# File: dataset.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import MIDI
from midi_tokenizer import MIDITokenizerV1, MIDITokenizerV2
from typing import Union

EXTENSION = [".mid", ".midi"]

def file_ext(fname):
    return os.path.splitext(fname)[1].lower()


def get_midi_list(path):
    all_files = {
        os.path.join(root, fname)
        for root, _dirs, files in os.walk(path)
        for fname in files
    }
    return sorted(f for f in all_files if file_ext(f) in EXTENSION)


class MidiDataset(Dataset):
    def __init__(self, midi_list, tokenizer: Union[MIDITokenizerV1, MIDITokenizerV2],
                 max_len=2048, min_file_size=3000, max_file_size=384000,
                 aug=True, check_quality=False, rand_start=True):
        self.tokenizer = tokenizer
        self.midi_list = midi_list
        self.max_len = max_len
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.aug = aug
        self.check_quality = check_quality
        self.rand_start = rand_start

    def __len__(self):
        return len(self.midi_list)

    def load_midi(self, index):
        path = self.midi_list[index]
        try:
            with open(path, 'rb') as f:
                datas = f.read()
            if len(datas) > self.max_file_size:
                raise ValueError("file too large")
            if len(datas) < self.min_file_size:
                raise ValueError("file too small")
            mid = MIDI.midi2score(datas)
            if max([0] + [len(track) for track in mid[1:]]) == 0:
                raise ValueError("empty track")
            mid = self.tokenizer.tokenize(mid)
            if self.check_quality and not self.tokenizer.check_quality(mid)[0]:
                raise ValueError("bad quality")
            if self.aug:
                mid = self.tokenizer.augment(mid)
        except Exception:
            return self.load_midi(random.randint(0, len(self) - 1))
        return mid

    def __getitem__(self, index):
        mid = np.asarray(self.load_midi(index), dtype=np.int16)
        if self.rand_start:
            start_idx = random.randrange(0, max(1, len(mid) - self.max_len))
            start_idx = random.choice([0, start_idx])
        else:
            max_start = max(1, len(mid) - self.max_len)
            start_idx = (index * (max_start // 8)) % max_start
        mid = mid[start_idx: start_idx + self.max_len].astype(np.int64)
        return torch.from_numpy(mid)

    def collate_fn(self, batch):
        max_len = max(len(mid) for mid in batch)
        padded = [F.pad(mid, (0, 0, 0, max_len - len(mid)), mode="constant",
                        value=self.tokenizer.pad_id) for mid in batch]
        return torch.stack(padded)
