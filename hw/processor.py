from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample

IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)


def _tile_grid(num_tiles: int) -> tuple[int, int]:
    if num_tiles <= 1:
        return 1, 1
    rows = int(math.floor(math.sqrt(num_tiles)))
    while rows > 1 and num_tiles % rows != 0:
        rows -= 1
    return rows, num_tiles // rows


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        size = int(self.config.image_size)
        num_tiles = max(1, int(self.config.num_tiles))
        rows, cols = _tile_grid(num_tiles)

        image = image.convert("RGB")

        if rows == 1 and cols == 1:
            tiles = [image.resize((size, size), Image.BICUBIC)]
        else:
            canvas = image.resize((cols * size, rows * size), Image.BICUBIC)
            tiles = []
            for i in range(rows):
                for j in range(cols):
                    box = (j * size, i * size, (j + 1) * size, (i + 1) * size)
                    tiles.append(canvas.crop(box))

        mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)

        tensors = []
        for tile in tiles:
            arr = np.asarray(tile, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            t = (t - mean) / std
            tensors.append(t)

        return torch.stack(tensors, dim=0)

    def _image_block(self) -> str:
        placeholders = " ".join([IMAGE_TOKEN] * int(self.config.num_image_tokens))
        return f"{IMAGE_START_TOKEN} {placeholders} {IMAGE_END_TOKEN}"

    @staticmethod
    def _assistant_answer(sample: MathVQASample) -> str:
        return sample.answer.strip()

    def _prompt_body(self, sample: MathVQASample) -> str:
        options_text = "\n".join(sample.options)
        return (
            "Реши визуально-математическую задачу. "
            "Выбери один вариант ответа и напиши только букву.\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            "Ответ:"
        )

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        prompt = f"{self._image_block()}\n{self._prompt_body(sample)}"
        if include_answer:
            prompt = f"{prompt} {self._assistant_answer(sample)}"
        return prompt

    def _encode_prompt_ids(self, sample: MathVQASample, reserve_for_answer: int = 0) -> list[int]:
        image_ids = list(self.tokenizer.encode(self._image_block(), add_special_tokens=False))
        body_ids = list(self.tokenizer.encode(self._prompt_body(sample), add_special_tokens=False))
        budget = max(0, int(self.config.max_length) - len(image_ids) - reserve_for_answer)
        return image_ids + body_ids[:budget]

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        answer_ids = list(self.tokenizer.encode(self._assistant_answer(sample), add_special_tokens=True))
        prompt_ids = self._encode_prompt_ids(sample, reserve_for_answer=len(answer_ids))

        input_ids = prompt_ids + answer_ids
        labels = [self.config.ignore_index] * len(prompt_ids) + list(answer_ids)
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def tokenize_for_generation(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        ids = self._encode_prompt_ids(sample, reserve_for_answer=0)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }

    def build_generation_inputs(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_for_generation(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        ignore_index = self.config.ignore_index
        has_labels = "labels" in batch[0]

        max_len = max(int(item["input_ids"].shape[0]) for item in batch)

        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad = max_len - int(ids.shape[0])
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad, dtype=mask.dtype)])
            input_ids.append(ids)
            attention_mask.append(mask)
            if has_labels:
                lab = item["labels"]
                if pad > 0:
                    lab = torch.cat([lab, torch.full((pad,), ignore_index, dtype=lab.dtype)])
                labels.append(lab)

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
        }
        if has_labels:
            out["labels"] = torch.stack(labels, dim=0)
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.stack([item["pixel_values"] for item in batch], dim=0)
        return out
