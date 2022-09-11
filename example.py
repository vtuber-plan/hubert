import torch, torchaudio

# Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()
pytorch_total_params = sum(p.numel() for p in hubert.parameters() if p.requires_grad)
print(f"{pytorch_total_params // 1e6}M")

# Load audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract speech units
units = hubert.units(x)