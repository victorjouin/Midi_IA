import os
import glob
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Dataset: raw byte-level MIDI ---
class SimpleMIDIDataset(Dataset):
    def __init__(self, data_dir, max_len=4096):
        self.files = [f for f in glob.glob(os.path.join(data_dir, "*.mid")) +
                      glob.glob(os.path.join(data_dir, "*.midi"))]
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path, 'rb') as f:
            data = f.read(self.max_len)
        # map bytes 0-255 to token IDs 1-256, reserve 0 for padding
        seq = torch.tensor([b + 1 for b in data], dtype=torch.long)
        if len(seq) < self.max_len:
            pad = torch.zeros(self.max_len - len(seq), dtype=torch.long)
            seq = torch.cat([seq, pad], dim=0)
        return seq

# simple collate: stack sequences
def collate_fn(batch):
    return torch.stack(batch)

# --- Model: causal Transformer ---
class SimpleTransformer(nn.Module):
    def __init__(self,
                 vocab_size=257,       # 256 byte values + padding
                 embed_dim=128,
                 nhead=1,
                 hidden_dim=512,
                 num_layers=4,
                 max_len=4096):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        b, seq_len = x.size()
        x = self.embed(x) + self.pos_embed[:, :seq_len, :]
        # causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        return self.fc_out(x)

# --- Generation utilities ---
def generate_sequence(model, device, initial_seq, gen_len, temperature):
    model.eval()
    seq = initial_seq
    with torch.no_grad():
        for _ in range(gen_len):
            inp = seq if seq.size(1) <= model.pos_embed.size(1) else seq[:, -model.pos_embed.size(1):]
            logits = model(inp)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)
    return seq.squeeze(0).cpu().tolist()


def save_midi_from_bytes(token_list, path):
    # convert token IDs back to bytes and write file
    data = bytes([max(0, min(255, t - 1)) for t in token_list])
    with open(path, 'wb') as f:
        f.write(data)

# --- Main: train or generate ---
def main():
    parser = argparse.ArgumentParser()
    # shared args
    parser.add_argument('--data_dir', type=str, default='data', help='Répertoire des fichiers MIDI')
    parser.add_argument('--max_len', type=int, default=4096, help='Longueur max des séquences')
    parser.add_argument('--checkpoint', type=str, default=None, help='Chemin vers le checkpoint (.pt)')
    # training args
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    # generation args
    parser.add_argument('--generate', action='store_true', help='Activer la génération')
    parser.add_argument('--generate_length', type=int, default=4096, help='Nombre de tokens à générer')
    parser.add_argument('--temperature', type=float, default=1.0, help='Température de génération')
    parser.add_argument('--output', type=str, default='output.mid', help='Chemin du fichier MIDI généré')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTransformer(
        vocab_size=257, embed_dim=128, nhead=1,
        hidden_dim=512, num_layers=4, max_len=args.max_len
    ).to(device)

    if args.generate:
        # 1) charger les poids
        assert args.checkpoint, '--checkpoint est requis pour la génération'
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        # 2) primer avec un vrai header MIDI
        header = [0x4D, 0x54, 0x68, 0x64, 0, 0, 0, 6,  0, 1, 0, 1, 1, 224]
        initial_seq = torch.tensor([b + 1 for b in header], dtype=torch.long, device=device)[None, :]
        gen = generate_sequence(model, device, initial_seq, args.generate_length, args.temperature)
        save_midi_from_bytes(gen, args.output)
        print(f'✅ Génération MIDI enregistrée dans {args.output}')
        return

    # --- TRAINING ---
    dataset = SimpleMIDIDataset(args.data_dir, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        # enumerate to avoid zero division
        for i, batch in enumerate(progress, start=1):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1), ignore_index=0
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=total_loss / i)
        print(f"Epoch {epoch}/{args.epochs} completed. Avg Loss: {total_loss/len(loader):.4f}")
    # sauvegarde du checkpoint
    save_path = args.checkpoint or 'checkpoint.pt'
    torch.save(model.state_dict(), save_path)
    print(f'✅ Poids enregistrés dans {save_path}')

if __name__ == '__main__':
    main()