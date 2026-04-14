# AL4FM Backend
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">

Server-side component of [AL4FM](../README.md). The backend runs on a GPU machine and exposes a FastAPI service that the annotation frontend talks to over HTTP (typically through an SSH tunnel). It wraps an object detector (RT-DETRv2 / RF-DETR) with the active learning logic, runs SAM-based mask extraction for new datasets, and serves the sampling queries that drive the UI.

## Status

Two pieces are not yet present in this public release and are being restored from the original lab checkout:

- `object_detectors/al_methods/FAST/sample_fast_main.py` — the FAST sampling routine that implements the active learning method described in the paper. It is imported at [`server.py:51`](server.py#L51) and used at [`server.py:513`](server.py#L513) and [`server.py:822`](server.py#L822). Until it is restored, the server will fail at import time.

The rest of the backend (training, evaluation, dataset sync, SAM mask extraction, API routing) is complete and runnable.

## Layout

```
backend/
├── server.py                        # FastAPI entry point (BackendServer class)
├── server_ml_functions/             # Thin ML glue used by server.py
│   ├── train_or_eval_od_model.py    #   - train_od_model / evaluate_od_model / extract_features_od_model
│   ├── get_masks_for_new_ids.py     #   - extract_gt_masks / extract_dense_masks
│   └── utils.py                     #   - load_annotations_for_ids etc.
└── object_detectors/
    ├── rt-detr/                     # Vendored RT-DETR / RT-DETRv2 (training configs + code)
    ├── rf-detr/                     # Vendored RF-DETR
    ├── al_methods/                  # Active learning samplers (FAST, …) — see status note above
    └── tools/                       # Offline preprocessing scripts and experiment notebooks
```

## Installation

```bash
cd backend
conda env create -f environment.yml
conda activate al4fm-backend
```

`environment.yml` installs PyTorch (CUDA 12.1), torchvision, FastAPI, rasterio, pycocotools and the rest of the scientific stack. The `pip:` block installs:

- `segment-geospatial` and `segment-anything-hq` — used as the SAM wrapper everywhere in the codebase.
- `faster-coco-eval` — faster replacement for `pycocotools` evaluation, required by the vendored RT-DETRv2 trainer.
- `autodistill` + `autodistill-grounded-sam` — **only** needed if you run `object_detectors/tools/extract_sam_box_masks.py` (Grounded-SAM box prompting). Drop them from `requirements.txt` if you are not using that script.

A CUDA-capable GPU with >= 8 GB VRAM is recommended. CPU-only operation is possible for the API surface but not for training or mask extraction.

## Running the Server

```bash
python server.py --port 8005 --cuda 0 --workdir server_workdir \
    --config object_detectors/rt-detr/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml
```

Notable command-line flags (see [`server.py`](server.py) for the full list):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--port` | `8005` | FastAPI port. Must match the SSH tunnel and the port entered in the frontend's startup wizard. |
| `--cuda` | `0` | CUDA device index. |
| `--workdir` | `server_workdir` | Directory where per-experiment weights, logs and the per-experiment `objects_al_gt.npy` are written. |
| `--config` | RT-DETRv2 config | Path to the detector training config. Ships with `rt-detr/configs/rtdetrv2/*.yml`; point this at whichever variant you want to train. |
| `--epochs`, `--batch_size`, `--learning_rate`, `--eval_every` | — | Training hyperparameters. |

On startup the server writes per-user activity logs to `logs/user_activity_log_<username>_localhost:<port>_<mode>.jsonl`.

## HTTP API

The frontend drives the server through the routes registered in `BackendServer.setup_routes` ([`server.py:104-146`](server.py#L104-L146)):

| Route | Method | Purpose |
| --- | --- | --- |
| `/get_next_active_image_ids` | POST | Initial sampling round (before any training has happened). |
| `/get_new_training_ids_smart` | GET | Active learning round using FAST sampling. |
| `/train_model` | POST | Train the detector on the current labeled set. |
| `/evaluate_model` | GET | Evaluate the detector on the training set. |
| `/extract_prototype_masks` | GET | Run GT-mask extraction for the current dataset. |
| `/extract_prototype_features` | GET | Cache detector features for the current dataset. |
| `/extract_dense_masks` | POST | Run dense SAM mask extraction on a dataset (invoked from the frontend's dataset wizard). |
| `/extraction_progress` | GET | Poll progress of a long-running extraction. |
| `/get_available_objects` | GET | List the categories available in the current dataset. |
| `/check_dataset_status`, `/prepare_dataset_sync`, `/upload_dataset_file`, `/finalize_dataset_sync` | — | Dataset sync endpoints used when the annotator uploads a new dataset from the frontend. |

## Dataset Layout

Datasets live under `datasets/<dataset_name>/`. The server expects a COCO-style annotation file plus 1024×1024 image tiles, and a set of `.npy` files produced by the SAM preprocessing step:

```
datasets/<dataset_name>/
├── images/
├── annotations.json
├── boxes_<dataset_name>_16.npy       # SAM box proposals
├── img_idx_<dataset_name>_16.npy     # per-proposal image index
└── objects_<dataset_name>_16.npy     # dense SAM masks
```

The `_16` suffix encodes the `--pixels_between_points` setting used when generating the dense masks. If any of these files are missing, smart sampling logs a warning and falls back to random sampling ([`server.py:495-505`](server.py#L495-L505)).

`objects_al_gt.npy` is written inside `--workdir` during an experiment and is not part of the dataset itself.

## Preprocessing Tools

`object_detectors/tools/` contains the scripts and notebooks used to prepare and analyse datasets:

| File | Purpose |
| --- | --- |
| `extract_sam_dense_masks.py` | Run SAM (HQ-SAM via `segment-geospatial`) to produce the `boxes_*.npy`, `img_idx_*.npy` and `objects_*.npy` files. |
| `extract_sam_box_masks.py` | Run Grounded-SAM to produce box-prompted masks. Only needed for the ablation where text prompts are used. |
| `benchmark_sam.py` | Time SAM across different point spacings. |
| `preprocess_fair1m.ipynb`, `preprocess_hrsc2016.ipynb`, `preprocess_fmow.ipynb`, `preprocess_xview.ipynb` | Notebook walkthroughs of how each public benchmark was converted into the AL4FM COCO layout. |
| `active_learning_schematic.ipynb`, `interactive_od_schematic.ipynb`, `interpretable_ai_schematic.ipynb`, `make_plots_attention.ipynb`, `plot_sam_results.ipynb`, `find_and_plot_overviews.ipynb`, `plot_examples_coco_style.ipynb` | Figure-generation notebooks used in the paper. |

Running `extract_sam_dense_masks.py` requires the dataset to already be registered in its `get_dataset_paths` helper; if you want to run it on a custom dataset, add an entry there or use the `/extract_dense_masks` API route instead (which takes arbitrary dataset names).

## Object Detectors

`object_detectors/rt-detr/` and `object_detectors/rf-detr/` are vendored copies of the upstream RT-DETR(v2) and RF-DETR repositories, with their own READMEs, configs and LICENSE files. The server currently drives RT-DETRv2 through `server_ml_functions/train_or_eval_od_model.py`.

## License

Released under the MIT license — see [LICENSE](../LICENSE).
