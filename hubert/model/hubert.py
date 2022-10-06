
from typing import Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from hubert.model.features import FeatureExtractor, FeatureProjection, PositionalConvEmbedding
from hubert.model.transformer import TransformerEncoder



class Hubert(nn.Module):
    def __init__(self, num_label_embeddings: int = 100,
                    extractor_hidden_size: int = 512,
                    hidden_size: int = 768,
                    feedforward_size: int = 3072,
                    transformer_nhead: int = 12,
                    transformer_layers: int = 12,
                    out_dim: int = 256,
                    mask: bool = True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor(extractor_hidden_size)
        self.feature_projection = FeatureProjection(extractor_hidden_size, hidden_size)
        self.positional_embedding = PositionalConvEmbedding(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, transformer_nhead, feedforward_size, activation="gelu", batch_first=True
            ),
            transformer_layers,
        )
        self.proj = nn.Linear(hidden_size, out_dim)

        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(hidden_size).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, out_dim)

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(
        self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.encode(x)
        x = self.proj(x)
        logits = self.logits(x)
        return logits, mask


class HubertSoft(Hubert):
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)


class HubertDiscrete(Hubert):
    def __init__(self, kmeans):
        super().__init__(504)
        self.kmeans = kmeans

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.LongTensor:
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav, layer=7)
        x = self.kmeans.predict(x.squeeze().cpu().numpy())
        return torch.tensor(x, dtype=torch.long, device=wav.device)

def _compute_mask(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask
