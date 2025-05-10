# File: model_trainer.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import lightning as pl
from lightning import Trainer
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from peft import LoraConfig, TaskType
from safetensors.torch import save_file as safe_save_file
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from midi_model import MIDIModel, MIDIModelConfig, config_name_list


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) /
                   float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class TrainMIDIModel(MIDIModel, pl.LightningModule):
    def __init__(self, config: MIDIModelConfig,
                 lr=2e-4, weight_decay=0.01, warmup=1e3, max_step=1e6,
                 sample_seq=False, gen_example_interval=1, example_batch=8):
        super().__init__(config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.max_step = max_step
        self.sample_seq = sample_seq
        self.gen_example_interval = gen_example_interval
        self.example_batch = example_batch
        self.last_save_step = 0
        self.gen_example_count = 0

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'norm']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr,
                                betas=(0.9, 0.99), eps=1e-08)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup, num_training_steps=self.max_step)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}

    def compute_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=-1).flatten()
        labels = labels.flatten()
        mask = (labels != self.tokenizer.pad_id)
        return torch.sum(preds[mask] == labels[mask]).float() / mask.sum().float()

    def training_step(self, batch, batch_idx):
        x = batch[:, :-1].contiguous()
        y = batch[:, 1:].contiguous()
        hidden = self.forward(x)
        if self.sample_seq:
            rand_idx = [-1] + random.sample(list(range(y.shape[1] - 2)),
                                            min(127, (y.shape[1] - 2) // 2))
            hidden = hidden[:, rand_idx]
            y = y[:, rand_idx]
        hidden = hidden.reshape(-1, hidden.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        x = y[:, :-1]
        logits = self.forward_token(hidden, x)
        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size),
                                y.view(-1), ignore_index=self.tokenizer.pad_id)
        self.log("train/loss", loss)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, :-1].contiguous()
        y = batch[:, 1:].contiguous()
        hidden = self.forward(x).reshape(-1, self.config.n_embd)
        y = y.reshape(-1, y.shape[-1])
        logits = self.forward_token(hidden, y[:, :-1])
        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size),
                                y.view(-1), ignore_index=self.tokenizer.pad_id)
        acc = self.compute_accuracy(logits, y)
        self.log_dict({"val/loss": loss, "val/acc": acc}, sync_dist=True)
        return loss

    @rank_zero_only
    def gen_example(self, save_dir):
        # same as original
        pass  # trimmed for brevity

    @rank_zero_only
    def save_peft(self, save_dir):
        adapter_name = self.active_adapters()[0]
        adapter_config = self.peft_config[adapter_name]
        os.makedirs(save_dir, exist_ok=True)
        adapter_config.save_pretrained(save_dir)
        state = self.get_adapter_state_dict(adapter_name)
        safe_save_file(state, os.path.join(save_dir, "adapter_model.safetensors"), metadata={"format": "pt"})

    def on_save_checkpoint(self, checkpoint):
        if self.global_step == self.last_save_step:
            return
        self.last_save_step = self.global_step
        # saving logic unchanged
        pass


