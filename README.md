# HuBERT

Training and inference scripts for the HuBERT content encoders.

## Example Usage

### Programmatic Usage

```python
import torch, torchaudio

# Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()

# Load audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract speech units
units = hubert.units(x)
```

### Script-Based Usage

```
usage: encode.py [-h] [--extension EXTENSION] {soft,discrete} in-dir out-dir

Encode an audio dataset.

positional arguments:
  {soft,discrete}       available models (HuBERT-Soft or HuBERT-Discrete)
  in-dir                path to the dataset directory.
  out-dir               path to the output directory.

optional arguments:
  -h, --help            show this help message and exit
  --extension EXTENSION
                        extension of the audio files (defaults to .flac).
```

## Training

### Step 1: Dataset Preparation

Download and extract the [LibriSpeech](https://www.openslr.org/12) corpus. The training script expects the following tree structure for the dataset directory:

```
│   lengths.json
│
└───wavs
    ├───dev-*
    │   ├───84
    │   ├───...
    │   └───8842
    └───train-*
        ├───19
        ├───...
        └───8975
```

The `train-*` and `dev-*` directories should contain the training and validation splits respectively. Note that there can be multiple `train` and `dev` folders e.g., `train-clean-100`, `train-other-500`, etc. Finally, the `lengths.json` file should contain key-value pairs with the file path and number of samples:

```json
{
    "dev-clean/1272/128104/1272-128104-0000": 93680,
    "dev-clean/1272/128104/1272-128104-0001": 77040,
}
```

### Step 2: Extract Discrete Speech Units

Encode LibriSpeech using the HuBERT-Discrete model and `encode.py` script:

```
usage: encode.py [-h] [--extension EXTENSION] {soft,discrete} in-dir out-dir

Encode an audio dataset.

positional arguments:
  {soft,discrete}       available models (HuBERT-Soft or HuBERT-Discrete)
  in-dir                path to the dataset directory.
  out-dir               path to the output directory.

optional arguments:
  -h, --help            show this help message and exit
  --extension EXTENSION
                        extension of the audio files (defaults to .flac).
```

for example:

```
python encode.py discrete path/to/LibriSpeech/wavs path/to/LibriSpeech/units
```

At this point the directory tree should look like:

```
│   lengths.json
│
├───discrete
│   ├───...
└───wavs
    ├───...
```

### Step 3: Train the HuBERT-Soft Content Encoder

```
usage: train.py [-h] [--resume RESUME] [--warmstart] [--mask] [--alpha ALPHA] dataset-dir checkpoint-dir

Train HuBERT soft content encoder.

positional arguments:
  dataset-dir      path to the data directory.
  checkpoint-dir   path to the checkpoint directory.

optional arguments:
  -h, --help       show this help message and exit
  --resume RESUME  path to the checkpoint to resume from.
  --warmstart      whether to initialize from the fairseq HuBERT checkpoint.
  --mask           whether to use input masking.
  --alpha ALPHA    weight for the masked loss.
```

## Links

- [Soft-VC repo](https://github.com/bshall/soft-vc)
- [Soft-VC paper](https://ieeexplore.ieee.org/abstract/document/9746484)
- [Official HuBERT repo](https://github.com/pytorch/fairseq)
- [HuBERT paper](https://arxiv.org/abs/2106.07447)

## Citation

If you found this work helpful please consider citing our paper:

```
@inproceedings{
    soft-vc-2022,
    author={van Niekerk, Benjamin and Carbonneau, Marc-André and Zaïdi, Julian and Baas, Matthew and Seuté, Hugo and Kamper, Herman},
    booktitle={ICASSP}, 
    title={A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion}, 
    year={2022}
}
```