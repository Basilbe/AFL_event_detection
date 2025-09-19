# AFL Event Detection Pipeline

This repository implements an end-to-end action recognition workflow for
Australian Rules Football clips. It provides:

* a from-scratch PyTorch implementation of the R(2+1)D network architecture
* data loaders that sample short clips from class-labelled gameplay footage
* training utilities for fine-tuning the model on small custom datasets
* an inference script that counts events in a new gameplay recording

The codebase is organised as a reusable Python package (`afl_event_detection`)
with dedicated modules for models, datasets, training, and inference helpers.

## Project structure

```
afl_event_detection/
├── data/                # dataset and dataloader helpers
├── models/              # custom R(2+1)D implementation
├── scripts/             # CLI entry points for training and inference
├── training/            # reusable training/evaluation loops
└── utils/               # transforms, video utilities, event aggregation
```

Two command-line interfaces drive the workflow:

* `python -m afl_event_detection.scripts.train` — fine-tune the model on your
  labelled clips.
* `python -m afl_event_detection.scripts.predict` — detect and count actions in
  an unseen video using a trained checkpoint.

## Preparing the dataset

Organise your curated 5–6 second clips into class-specific folders:

```
my_clips/
├── kick/
│   ├── kick_01.mp4
│   ├── kick_02.mp4
│   └── ...
├── mark/
│   ├── mark_01.mp4
│   └── ...
└── tackle/
    ├── tackle_01.mp4
    └── ...
```

The loader will automatically discover the class names and sample fixed-length
clips from each video. Additional classes (e.g. handball) can be added by
creating new folders.

## Training a model

```
python -m afl_event_detection.scripts.train \
    my_clips \
    outputs/experiment_01 \
    --epochs 40 \
    --batch-size 4 \
    --clip-length 16 \
    --step-between-clips 4 \
    --resize 128 \
    --crop-size 112
```

Important options:

* `--clip-length` — number of frames per training clip.
* `--step-between-clips` — temporal stride when sampling clips from a video.
* `--resize` / `--crop-size` — spatial pre-processing.
* `--no-amp` — disable mixed precision if you prefer full precision training.

Every epoch prints training/validation statistics. The best performing weights
(on the validation split) are saved to `<output_dir>/best.pt` alongside a
`training_summary.json` with final metrics.

## Running inference on new footage

```
python -m afl_event_detection.scripts.predict \
    outputs/experiment_01/best.pt \
    gameplay_snippet.mp4 \
    --window-stride 8 \
    --min-confidence 0.6 \
    --output-json predictions.json
```

The inference script applies the same pre-processing captured in the
checkpoint, slides a window across the gameplay clip, and aggregates confident
predictions into discrete events. The JSON output contains per-class counts and
frame ranges (start/end) for each detected action.

## Requirements

* Python 3.10+
* PyTorch 2.0+
* TorchVision 0.15+

Install dependencies via pip:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA version as needed
```

## Notebook reference

The original experimentation notebook (`01_model_train.ipynb`) is included for
reference, but the primary workflow is driven by the scripts described above.
