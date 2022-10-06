import torch, torchaudio
import librosa
import soundfile as sf

import os
import traceback

def resample(file: str, out: str, target_sr: int = 16000):
    try:
        y, sr = librosa.load(file, sr=None)
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sf.write(out, y_16k, target_sr, 'PCM_24')
    except Exception as e:
        traceback.print_stack()


# Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()
pytorch_total_params = sum(p.numel() for p in hubert.parameters() if p.requires_grad)
print(f"{pytorch_total_params // 1e6}M")

# Load audio
resample("LJ011-0110.wav", "LJ011-0110_16k.wav", 16000)
wav, sr = torchaudio.load("LJ011-0110_16k.wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract speech units
units = hubert.units(wav)

print(units.shape)