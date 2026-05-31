from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        self.norm = nn.LayerNorm(vision_hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(vision_hidden_states)
        x = x.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, self.num_image_tokens)
        x = x.transpose(1, 2)
        return self.proj(x)


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    merged = input_embeds.clone()
    mask = input_ids == image_token_id
    merged[mask] = visual_embeds.reshape(-1, visual_embeds.shape[-1]).to(merged.dtype)
    return merged


class MathVLM(nn.Module):
    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, t = pixel_values.shape[0], pixel_values.shape[1]
        flat = pixel_values.flatten(0, 1)
        out = self.vision_encoder(flat)
        hidden = getattr(out, "last_hidden_state", out)
        return hidden.reshape(b, t * hidden.shape[1], hidden.shape[2])

    def _visual_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.encode_images(pixel_values))

    def _merged_embeds(self, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        visual_embeds = self._visual_embeds(pixel_values).to(text_embeds.dtype)
        return merge_visual_embeddings(text_embeds, input_ids, visual_embeds, self.config.image_token_id)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        inputs_embeds = self._merged_embeds(batch["input_ids"], batch["pixel_values"])
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        inputs_embeds = self._merged_embeds(batch["input_ids"], batch["pixel_values"])
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            **generation_kwargs,
        )
