# File: train.py
#!/usr/bin/env python3
import argparse
import os
import random
import torch
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset import get_midi_list, MidiDataset
from model_trainer import TrainMIDIModel
from midi_model import MIDIModelConfig, config_name_list
from midi_tokenizer import MIDITokenizer


def limit_gpu_memory(target_gb=22, device=0):
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory
        fraction = min(1.0, target_gb * 1024**3 / total)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        print(f"Limiting GPU memory to {target_gb} GB (fraction={fraction:.2f})")


def main():
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--resume", type=str, default="", help="resume training from ckpt")
    parser.add_argument("--ckpt", type=str, default="", help="load ckpt")
    parser.add_argument("--config", type=str, default="tv2o-medium", help="model config name or file")
    parser.add_argument("--task", type=str, default="train", choices=["train", "lora"], help="Full train or lora")
    # dataset args
    parser.add_argument("--data", type=str, default="data", help="dataset path")
    parser.add_argument("--data-val-split", type=int, default=128,
                        help="the number of midi files divided into the validation set")
    parser.add_argument("--max-len", type=int, default=512, help="max seq length for training")
    parser.add_argument("--quality", action="store_true", default=False, help="check dataset quality")
    # training args
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--warmup-step", type=float, default=1e2, help="warmup step")
    parser.add_argument("--max-step", type=int, default=1e6, help="max training step")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient clip val")
    parser.add_argument("--sample-seq", action="store_true", default=False, help="sample midi seq to reduce vram")
    parser.add_argument("--gen-example-interval", type=int, default=1, help="generate example interval. set 0 to disable")
    parser.add_argument("--batch-size-train", type=int, default=4, help="batch size for training")
    parser.add_argument("--batch-size-val", type=int, default=4, help="batch size for val")
    parser.add_argument("--batch-size-gen-example", type=int, default=8, help="batch size for generate example")
    parser.add_argument("--workers-train", type=int, default=4, help="workers num for training dataloader")
    parser.add_argument("--workers-val", type=int, default=4, help="workers num for validation dataloader")
    parser.add_argument("--acc-grad", type=int, default=2, help="gradient accumulation")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "auto"], help="accelerator")
    parser.add_argument("--precision", type=str, default="bf16-true", choices=["16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true", "64", "32", "16", "bf16"], help="precision")
    parser.add_argument("--devices", type=int, default=-1, help="devices num")
    parser.add_argument("--nodes", type=int, default=1, help="nodes num")
    parser.add_argument("--disable-benchmark", action="store_true", default=False,help="disable cudnn benchmark")
    parser.add_argument("--log-step", type=int, default=1, help="log training loss every n steps")
    parser.add_argument("--val-step", type=int, default=50, help="valid and save every n steps, set 0 to valid and save every epoch")

    opt = parser.parse_args()
    print(opt)

    limit_gpu_memory(20, 0)
    if opt.disable_benchmark:
        torch.backends.cudnn.benchmark = False

    if not os.path.exists("lightning_logs"):
        os.mkdir("lightning_logs")
    if not os.path.exists("sample"):
        os.mkdir("sample")

    pl.seed_everything(opt.seed)

    if opt.config in config_name_list:
        config = MIDIModelConfig.from_name(opt.config)
    else:
        config = MIDIModelConfig.from_json_file(opt.config)
    tokenizer = config.tokenizer

    midi_list = get_midi_list(opt.data)
    random.shuffle(midi_list)
    train_list = midi_list[:-opt.data_val_split]
    val_list = midi_list[-opt.data_val_split:]

    train_ds = MidiDataset(train_list, tokenizer, max_len=opt.max_len,
                            aug=True, check_quality=opt.quality, rand_start=True)
    val_ds = MidiDataset(val_list, tokenizer, max_len=opt.max_len,
                          aug=False, check_quality=opt.quality, rand_start=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=opt.batch_size_train,
                                               shuffle=True, num_workers=opt.workers_train,
                                               pin_memory=True, collate_fn=train_ds.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=opt.batch_size_val,
                                             shuffle=False, num_workers=opt.workers_val,
                                             pin_memory=True, collate_fn=val_ds.collate_fn)

    model = TrainMIDIModel(config, lr=opt.lr, weight_decay=opt.weight_decay,
                           warmup=opt.warmup_step, max_step=opt.max_step,
                           sample_seq=opt.sample_seq,
                           gen_example_interval=opt.gen_example_interval,
                           example_batch=opt.batch_size_gen_example)

    if opt.ckpt:
        ckpt = torch.load(opt.ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    elif opt.task == "lora":
        raise ValueError("--ckpt must be set to train lora")
    if opt.task == "lora":
        model.requires_grad_(False)
        lora_cfg = LoraConfig(r=64, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                               task_type=TaskType.CAUSAL_LM, bias="none",
                               lora_alpha=128, lora_dropout=0)
        model.add_adapter(lora_cfg)

    ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1,
                              save_last=True, auto_insert_metric_name=False,
                              filename="epoch={epoch},loss={val/loss:.4f}")

    trainer = Trainer(  precision=opt.precision,
                        accumulate_grad_batches=opt.acc_grad,
                      gradient_clip_val=opt.grad_clip,
                      accelerator=opt.accelerator,
                      devices=opt.devices,
                      num_nodes=opt.nodes,
                      max_steps=opt.max_step,
                      benchmark=not opt.disable_benchmark,
                      val_check_interval=opt.val_step or None,
                      limit_val_batches=0.0,
                      log_every_n_steps=opt.log_step,
                      strategy="auto",
                      callbacks=[ckpt_cb])

    trainer.fit(model, train_loader, val_loader, ckpt_path=opt.resume or None)

if __name__ == '__main__':
    main()
