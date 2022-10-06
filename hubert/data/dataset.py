import os
import random
from pathlib import Path
import numpy as np
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio


class AcousticUnitsDataset(Dataset):
    def __init__(
        self,
        root: Path,
        sample_rate: int = 16000,
        label_rate: int = 50,
        min_samples: int = 32000,
        max_samples: int = 250000
    ):
        self.filepaths = []
        self.lengths = []
        with open(root, "r", encoding="utf-8") as f:
            self.root_path = f.readline().rstrip()
            for line in f:
                if len(line.strip()) == 0:
                    continue
                p, l = line.rstrip().split()
                self.filepaths.append(p)
                self.lengths.append(int(l))

        new_filepaths = []
        new_lengths = []
        for i, p in enumerate(self.filepaths):
            abs_path = os.path.join(self.root_path, p).replace("\\", "/")
            if self.lengths[i] < min_samples or self.lengths[i] > max_samples:
                continue
            new_filepaths.append(p)
            new_lengths.append(self.lengths[i])

        self.filepaths = new_filepaths
        self.lengths = new_lengths

        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.min_samples = min_samples
        self.max_samples = max_samples

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        wav_path = self.filepaths[index]
        units_path = self.units_dir / wav_path.relative_to(self.wavs_dir)

        wav, _ = torchaudio.load(wav_path)
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        codes = np.load(units_path.with_suffix(".npy"))

        return wav, torch.from_numpy(codes).long()

    def collate(self, batch):
        wavs, codes = zip(*batch)
        wavs, codes = list(wavs), list(codes)

        wav_lengths = [wav.size(-1) for wav in wavs]
        code_lengths = [code.size(-1) for code in codes]

        wav_frames = min(self.max_samples, *wav_lengths)

        collated_wavs, wav_offsets = [], []
        for wav in wavs:
            wav_diff = wav.size(-1) - wav_frames
            wav_offset = random.randint(0, wav_diff)
            wav = wav[:, wav_offset : wav_offset + wav_frames]

            collated_wavs.append(wav)
            wav_offsets.append(wav_offset)

        rate = self.label_rate / self.sample_rate
        code_offsets = [round(wav_offset * rate) for wav_offset in wav_offsets]
        code_frames = round(wav_frames * rate)
        remaining_code_frames = [
            length - offset for length, offset in zip(code_lengths, code_offsets)
        ]
        code_frames = min(code_frames, *remaining_code_frames)

        collated_codes = []
        for code, code_offset in zip(codes, code_offsets):
            code = code[code_offset : code_offset + code_frames]
            collated_codes.append(code)

        wavs = torch.stack(collated_wavs, dim=0)
        codes = torch.stack(collated_codes, dim=0)

        return wavs, codes
