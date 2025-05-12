import os
import argparse
import torch
from pathlib import Path
import MIDI
from midi_model import MIDIModelConfig, config_name_list
from train import TrainMIDIModel

def find_latest_checkpoint(log_dir):
    logs = Path(log_dir)
    versions = list(logs.glob('version_*'))
    if not versions:
        raise FileNotFoundError(f"No version dirs in {log_dir}")
    latest = max(versions, key=lambda d: int(d.name.split('_')[1]))
    ckpt_dir = latest / 'checkpoints'
    last_ckpt = ckpt_dir / 'last.ckpt'
    if last_ckpt.exists():
        return str(last_ckpt)
    all_ckpts = list(ckpt_dir.glob('*.ckpt'))
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return str(max(all_ckpts, key=lambda f: f.stat().st_mtime))

def main():
    parser = argparse.ArgumentParser(description="Generate and save a MIDI file from the trained model")
    parser.add_argument('--log_dir', type=str, default='lightning_logs', help='Root lightning logs dir')
    parser.add_argument('--ckpt', type=str, default='', help='Specific checkpoint path (overrides log_dir)')
    parser.add_argument('--config', type=str, default='tv2o-medium', help='Model config name or file')
    parser.add_argument('--output', type=str, default='output.mid', help='Output MIDI file path')
    parser.add_argument('--max_len', type=int, default=512, help='Max generation length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.98, help='Top-p sampling')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k sampling')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    opt = parser.parse_args()

    # Determine checkpoint path
    ckpt_path = opt.ckpt or find_latest_checkpoint(opt.log_dir)
    print(f"Using checkpoint: {ckpt_path}")

    # Load model config
    if opt.config in config_name_list:
        config = MIDIModelConfig.from_name(opt.config)
    else:
        config = MIDIModelConfig.from_json_file(opt.config)

    # Initialize and load model
    model = TrainMIDIModel(config)
    ckpt_data = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt_data.get('state_dict', ckpt_data)
    # Strip Lightning 'model.' prefix if present
    new_sd = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # Optional: set generator for reproducibility
    generator = None
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(opt.seed)

    # Generate token sequence
    token_tensor = model.generate(
        prompt=None,
        batch_size=1,
        max_len=opt.max_len,
        temp=opt.temperature,
        top_p=opt.top_p,
        top_k=opt.top_k,
        generator=generator
    )  # returns tensor of shape (1, L)
    tokens = token_tensor.squeeze(0).tolist()

    # Detokenize to score and convert to MIDI bytes
    midi_score = model.tokenizer.detokenize(tokens)
    midi_bytes = MIDI.score2midi(midi_score)
    with open(opt.output, 'wb') as f:
        f.write(midi_bytes)
    print(f"Saved generated MIDI to {opt.output}")

if __name__ == '__main__':
    main()