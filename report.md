1# Report

## Track

Выбранный трек:

```text
A (CPU-only)
```

Обязательная часть: реализован весь инженерный пайплайн, пройдены public tests,
обучение и benchmark запущены локально на CPU без GPU и без скачивания внешних
датасетов. Внешние чекпойнты (`google/vit-base-patch16-224`, `Qwen/Qwen2-0.5B`)
для Track B/C подключаются через `build_hf_vlm` в `hw/backbones.py`, но на CPU не
запускались.

## Что реализовано

- [x] dataset.py — чтение `manifest.jsonl`, фильтрация по `split` и (опционально)
  `subject`, `max_samples`, разрешение пути к картинке относительно манифеста,
  открытие в RGB, `sanitize_question`, сборка `MathVQASample`.
- [x] processor.py — приведение к RGB + resize, тайлинг (`num_tiles` → near-square
  grid), нормализация (mean/std = 0.5), prompt с `<image_start> <image>×K
  <image_end>`, токенизация с маскированием prompt через `IGNORE_INDEX` (loss
  только на ответе + eos), image-aware усечение (image-блок никогда не режется),
  `collate` с паддингом и `tokenize_for_generation`.
- [x] model.py — `VisionToTextAdapter` (LayerNorm → adaptive_avg_pool1d к
  `num_image_tokens` → Linear→GELU→Linear), `merge_visual_embeddings` (вставка
  visual-эмбеддингов на позиции `<image>`), `MathVLM.encode_images` (склейка
  тайлов по seq-оси), `forward` (loss) и `generate`, заморозка backbone.
- [x] train.py — `train_one_step` (поддерживает dict- и attribute-loss, проверка
  finite, zero_grad→backward→step), `run_training` с gradient accumulation
  (`global_batch_size // local_batch_size`), `fast_train`, AdamW по обучаемым
  параметрам (адаптер), сохранение адаптера.
- [x] benchmark.py — `parse_mc_answer` (A / (B) / "Answer: C" / "... is D." /
  None; регистронезависимый fallback), `build_benchmark_prompt`,
  `compute_accuracy` (overall + by subject), `run_benchmark` (generate → parse →
  метрики, запись предсказаний).
- [x] (доп.) backbones.py — `SimpleTokenizer` + крошечные mock vision/LLM и
  фабрика `build_vlm`, чтобы весь пайплайн запускался на CPU (соответствует
  `tiny/local-or-mocked` в конфиге Track A).

## Конфигурация

```text
config path: configs/track_a_cpu.yaml
seed:        42
device:      cpu (torch.cuda.is_available() == False)
dtype:       float32
max_steps:   3 (с --fast-train: 2)
batch size:  local_batch_size=1, global_batch_size=1  -> grad accumulation steps = 1
```

Окружение: Python 3.12.12, torch 2.12.0, numpy 2.4.6, pillow 12.2.0,
платформа macOS arm64 (Apple Silicon).

## Результаты

```text
public tests:        14 passed  (python -m pip install -e ".[dev]"; pytest -q tests_public)

train loss (configs/track_a_cpu.yaml, seed 42, CPU, обучаемых параметров адаптера = 24 960):
  step 1/3  loss = 4.7534
  step 2/3  loss = 4.7668
  step 3/3  loss = 4.6880   (final)
  loss конечный на всех шагах (проверяется assert torch.isfinite)

benchmark accuracy (python -m hw.benchmark --config configs/inference_math.yaml --toy, seed 42, воспроизводимо):
  split = dev, 4 примера toy-набора
  overall            = 0.0
  subject/geometry   = 0.0
  subject/plots      = 0.0
```

Важно: toy-benchmark прогонялся на **необученном, случайно инициализированном
mock-бэкенде** (Track A не требует обучения VLM до качества). Согласно README,
toy-набор — это smoke-check работоспособности пайплайна, а не метрика качества,
поэтому accuracy 0.0 здесь ожидаема и не характеризует модель. Содержательная
оценка качества — на MathVista testmini (расширенный трек, не запускалась).

## Использованные ресурсы

```text
CPU/GPU:          только CPU (Apple Silicon arm64), GPU/CUDA не использовался
VRAM:             0 ГБ (CPU)
время обучения:   ~несколько секунд на 3 шага; public tests ~1 c
```

## Анализ ошибок

Наблюдения по фактическому прогону `--toy` (4 dev-примера, `artifacts/predictions.jsonl`).
Это режим smoke-check со случайно инициализированным mock-бэкендом без обучения.

1. **Нет визуальной привязки / вырожденная генерация.** На всех 4 разных
   изображениях и вопросах модель выдала один и тот же текст
   (`"и 180° Варианты: равна На 11 только A) 90° 10 Реши 13 25 ... D) Прямоуг..."`).
   Случайный адаптер не переносит признаки картинки в пространство LLM, поэтому
   выход не зависит ни от изображения, ни от вопроса.
2. **Коллапс к одному варианту.** `parse_mc_answer` извлекает из вывода букву
   `D` для всех 4 примеров, то есть модель отвечает одинаково независимо от
   вопроса — реального выбора между вариантами нет.
3. **Не читает числовые значения с графиков/фигур.** Для geometry (gold C, B) и
   для plots (gold C) предсказание `D` неверно во всех случаях (0/4): крошечный
   (64-dim conv + 128-dim 1-слойный LM) необученный бэкенд не способен к
   визуально-математическому reasoning.

Все три — закономерные провалы необученного smoke-бэкенда; для содержательного
анализа ошибок нужен обученный адаптер (Track B/C) и оценка на MathVista.

## Комментарии

Самое сложное:
- согласовать число visual-токенов между prompt и моделью: vision-энкодер выдаёт
  произвольное число патчей, а на позициях `<image>` их ровно `num_image_tokens`;
  решено `adaptive_avg_pool1d` в адаптере до фиксированного `num_image_tokens`;
- корректное маскирование `labels` (loss только на ответе + eos) и при этом
  гарантированно не «срезать» image-блок при усечении по `max_length`;
- сделать пайплайн полностью запускаемым на CPU без внешних чекпойнтов — для
  этого добавлен self-contained mock-бэкенд (`hw/backbones.py`), что и позволило
  получить реальные числа (loss, accuracy) без GPU.

Что бы улучшил при наличии ресурсов:
- Track B/C: обучить адаптер на `google/vit-base-patch16-224` + `Qwen2-0.5B-Instruct`
  на medium-наборе, затем SFT с LoRA;
- прогнать quality-evaluation на MathVista testmini и сравнить с baseline;
- усилить resampler (cross-attention/Perceiver вместо average pooling) и добавить
  настоящий tile-overlap в препроцессинг изображений.

## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
