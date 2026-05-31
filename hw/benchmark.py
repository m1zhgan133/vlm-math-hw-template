from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    if not text:
        return None
    valid = set(choices)

    marker = re.search(
        r"(?:answer|ответ)\s*(?:is|:|-|—|=)?\s*\(?\s*([A-Ea-e])\b",
        text,
        flags=re.IGNORECASE,
    )
    if marker and marker.group(1).upper() in valid:
        return marker.group(1).upper()

    candidates = [
        c.upper()
        for c in re.findall(r"(?<![A-Za-z])([A-Ea-e])(?![A-Za-z])", text)
        if c.upper() in valid
    ]
    if candidates:
        return candidates[-1]
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def _load_adapter(model: Any, adapter_path: str) -> None:
    path = Path(adapter_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(str(path), map_location="cpu")
    model.adapter.load_state_dict(state)


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    from hw.backbones import build_vlm
    from hw.dataset import MathVQADataset

    torch.manual_seed(int(config.get("seed", 42)))

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    infer_cfg = config.get("inference", {})

    manifest = data_cfg.get("eval_manifest")
    split = data_cfg.get("split", "dev")
    if toy:
        manifest = manifest or "assets/toy_math_vqa/manifest.jsonl"
    dataset = MathVQADataset(manifest, split=split, max_samples=data_cfg.get("max_samples"))

    device = torch.device(infer_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[benchmark] CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    model, processor, tokenizer = build_vlm(config, corpus=[dataset], force_mock=toy)

    adapter_path = model_cfg.get("adapter_path")
    if adapter_path and not toy and Path(adapter_path).exists():
        _load_adapter(model, adapter_path)
        print(f"[benchmark] loaded adapter from {adapter_path}")

    model.to(device)
    model.eval()

    max_new_tokens = int(infer_cfg.get("max_new_tokens", 16))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    rows: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        item = processor.build_generation_inputs(sample)
        batch = processor.collate([item])
        batch = {k: v.to(device) for k, v in batch.items()}

        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id
        generated = model.generate(batch, **gen_kwargs)

        text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        prediction = parse_mc_answer(text)
        rows.append(
            {
                "id": sample.id,
                "subject": sample.subject,
                "question": sample.question,
                "answer": sample.answer,
                "raw_output": text,
                "prediction": prediction,
            }
        )

    output_path = infer_cfg.get("output_path")
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[benchmark] wrote {len(rows)} predictions to {output_path}")

    metrics = compute_accuracy(rows)
    metrics["num_samples"] = float(len(rows))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
