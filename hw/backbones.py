from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig

SPECIAL_TOKENS = ("<pad>", "<eos>", IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN, "<unk>")


class SimpleTokenizer:
    def __init__(self) -> None:
        self.id2tok: list[str] = []
        self.tok2id: dict[str, int] = {}
        self.frozen = False

        for tok in SPECIAL_TOKENS:
            self._add(tok)

        self.pad_token_id = self.tok2id["<pad>"]
        self.eos_token_id = self.tok2id["<eos>"]
        self.unk_token_id = self.tok2id["<unk>"]

    def _add(self, tok: str) -> int:
        if tok not in self.tok2id:
            self.tok2id[tok] = len(self.id2tok)
            self.id2tok.append(tok)
        return self.tok2id[tok]

    def __len__(self) -> int:
        return len(self.id2tok)

    @staticmethod
    def _split(text: str) -> list[str]:
        return text.replace("\n", " ").split()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for tok in self._split(text):
            if tok in self.tok2id:
                ids.append(self.tok2id[tok])
            elif not self.frozen:
                ids.append(self._add(tok))
            else:
                ids.append(self.unk_token_id)
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tok2id.get(token, self.unk_token_id)

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        specials = set(SPECIAL_TOKENS)
        toks: list[str] = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.id2tok):
                continue
            tok = self.id2tok[i]
            if skip_special_tokens and tok in specials:
                continue
            toks.append(tok)
        return " ".join(toks)

    def freeze(self) -> None:
        self.frozen = True


class MockVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int = 64, image_size: int = 224, patch_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch(pixel_values)
        return x.flatten(2).transpose(1, 2)


class MockLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        max_position: int = 2048,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)
        self.ignore_index = ignore_index

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Embedding(max_position, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def _backbone(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        b, length, _ = inputs_embeds.shape
        pos_ids = torch.arange(length, device=inputs_embeds.device)
        x = inputs_embeds + self.pos(pos_ids).unsqueeze(0)

        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
        )
        key_padding = None
        if attention_mask is not None:
            key_padding = attention_mask == 0

        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h, attn_mask=causal, key_padding_mask=key_padding, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return self.ln_f(x)

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        x = self._backbone(inputs_embeds, attention_mask)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.ignore_index,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 16,
        eos_token_id: int = 1,
        **_: Any,
    ) -> torch.Tensor:
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        cur, cur_mask = inputs_embeds, attention_mask

        generated: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _step in range(int(max_new_tokens)):
            x = self._backbone(cur, cur_mask)
            next_id = self.lm_head(x[:, -1, :]).argmax(dim=-1)
            generated.append(next_id)

            finished = finished | (next_id == eos_token_id)
            if bool(finished.all()):
                break

            cur = torch.cat([cur, self.embed(next_id).unsqueeze(1)], dim=1)
            if cur_mask is not None:
                ones = torch.ones(batch_size, 1, dtype=cur_mask.dtype, device=device)
                cur_mask = torch.cat([cur_mask, ones], dim=1)

        if not generated:
            return torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        return torch.stack(generated, dim=1)


def make_processor_config(proc_cfg: dict[str, Any]) -> ProcessorConfig:
    return ProcessorConfig(
        image_size=int(proc_cfg.get("image_size", 224)),
        num_tiles=int(proc_cfg.get("num_tiles", 1)),
        tile_overlap=float(proc_cfg.get("tile_overlap", 0.0)),
        num_image_tokens=int(proc_cfg.get("num_image_tokens", 49)),
        max_length=int(proc_cfg.get("max_length", 512)),
        ignore_index=int(proc_cfg.get("ignore_index", -100)),
    )


def _is_mock_name(name: str | None) -> bool:
    if not name:
        return True
    name = name.lower()
    return any(key in name for key in ("tiny", "mock", "local-or-mocked"))


def build_mock_vlm(
    model_cfg: dict[str, Any],
    proc_cfg: ProcessorConfig,
    corpus: Iterable[MathVQADataset] = (),
) -> tuple[MathVLM, MathVLMProcessor, SimpleTokenizer]:
    tokenizer = SimpleTokenizer()
    processor = MathVLMProcessor(tokenizer, proc_cfg)

    for dataset in corpus:
        for idx in range(len(dataset)):
            processor.tokenize_sample(dataset[idx])
    tokenizer.freeze()

    vision_hidden = int(model_cfg.get("vision_hidden_size", 64))
    text_hidden = int(model_cfg.get("text_hidden_size", 128))
    patch_size = int(model_cfg.get("patch_size", 32))

    vision = MockVisionEncoder(vision_hidden, proc_cfg.image_size, patch_size)
    language = MockLanguageModel(
        len(tokenizer), hidden_size=text_hidden, ignore_index=proc_cfg.ignore_index
    )
    config = ModelConfig(
        vision_hidden_size=vision_hidden,
        text_hidden_size=text_hidden,
        num_image_tokens=proc_cfg.num_image_tokens,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
    )
    model = MathVLM(vision, language, config)
    return model, processor, tokenizer


def build_hf_vlm(
    model_cfg: dict[str, Any], proc_cfg: ProcessorConfig
) -> tuple[MathVLM, MathVLMProcessor, Any]:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["language_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
    )

    vision = AutoModel.from_pretrained(model_cfg["vision_encoder"])
    language = AutoModelForCausalLM.from_pretrained(model_cfg["language_model"])
    language.resize_token_embeddings(len(tokenizer))

    config = ModelConfig(
        vision_hidden_size=int(vision.config.hidden_size),
        text_hidden_size=int(language.config.hidden_size),
        num_image_tokens=proc_cfg.num_image_tokens,
        image_token_id=int(tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)),
    )
    processor = MathVLMProcessor(tokenizer, proc_cfg)
    model = MathVLM(vision, language, config)
    return model, processor, tokenizer


def build_vlm(
    config: dict[str, Any],
    corpus: Iterable[MathVQADataset] = (),
    force_mock: bool = False,
) -> tuple[MathVLM, MathVLMProcessor, Any]:
    model_cfg = dict(config.get("model", {}))
    proc_cfg = make_processor_config(config.get("processor", {}))

    use_mock = (
        force_mock
        or _is_mock_name(model_cfg.get("vision_encoder"))
        or _is_mock_name(model_cfg.get("language_model"))
    )
    if not use_mock:
        return build_hf_vlm(model_cfg, proc_cfg)
    return build_mock_vlm(model_cfg, proc_cfg, corpus)
