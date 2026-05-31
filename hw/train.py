from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from hw.backbones import build_vlm
from hw.dataset import MathVQADataset


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_loss(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return output["loss"]
    return output.loss


def train_one_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()

    output = model(batch)
    loss = _extract_loss(output)
    if loss is None or not torch.isfinite(loss):
        raise ValueError("Loss is None or non-finite")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.detach())


class _ProcessedDataset(Dataset):
    def __init__(self, base: MathVQADataset, processor: Any) -> None:
        self.base = base
        self.processor = processor

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.processor(self.base[idx])


def _save_adapter(model: torch.nn.Module, save_path: str | Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.adapter.state_dict().items()}

    if save_path.suffix == ".safetensors":
        try:
            from safetensors.torch import save_file

            save_file(state, str(save_path))
            return save_path
        except Exception:
            save_path = save_path.with_suffix(".pt")

    torch.save(state, str(save_path))
    return save_path


def run_training(config: dict[str, Any], fast_train: bool = False) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    trainer_cfg = config.get("trainer", {})

    device = torch.device(trainer_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    base_dataset = MathVQADataset(
        data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )
    model, processor, _tokenizer = build_vlm(config, corpus=[base_dataset])

    if model_cfg.get("freeze_vision", True):
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
    if model_cfg.get("freeze_llm", True):
        for p in model.language_model.parameters():
            p.requires_grad = False

    model.to(device)

    local_bs = int(trainer_cfg.get("local_batch_size", 1))
    global_bs = int(trainer_cfg.get("global_batch_size", local_bs))
    accum_steps = max(1, global_bs // max(1, local_bs))

    dataset = _ProcessedDataset(base_dataset, processor)
    loader = DataLoader(
        dataset,
        batch_size=local_bs,
        shuffle=True,
        collate_fn=processor.collate,
        num_workers=int(trainer_cfg.get("num_workers", 0)),
        drop_last=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(trainer_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] device={device} trainable_params={n_trainable} accum_steps={accum_steps}")

    max_steps = int(trainer_cfg.get("max_steps", 100))
    if fast_train:
        max_steps = min(max_steps, 2)
    num_epochs = int(trainer_cfg.get("num_train_epochs", 1))

    model.train()
    optimizer.zero_grad()

    losses: list[float] = []
    step = 0
    micro = 0
    stop = False

    for _epoch in range(num_epochs):
        if stop:
            break
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch)
            loss = _extract_loss(output)
            if loss is None or not torch.isfinite(loss):
                raise ValueError("Non-finite loss encountered during training")

            (loss / accum_steps).backward()
            micro += 1

            if micro % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                losses.append(float(loss.detach()))
                print(f"[train] step {step}/{max_steps} loss={losses[-1]:.4f}")
                if step >= max_steps:
                    stop = True
                    break

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        written = _save_adapter(model, save_path)
        print(f"[train] saved adapter to {written}")

    summary = {
        "steps": step,
        "losses": losses,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "trainable_params": n_trainable,
    }
    print(f"[train] done: {summary['steps']} steps, final_loss={summary['final_loss']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
