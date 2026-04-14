# Active Learning Meets Foundation Models (AL4FM)

This is the repository accompanying the "Active Learning Meets Foundation Models: Fast Remote Sensing Data Annotation for Object Detection" paper accepted at ICCV 2025. The code is released under the MIT license.

Active Learning Meets Foundation Models (AL4FM) is a real-time active learning and semi-automated labeling framework that leverages foundation models to streamline dataset annotation for object detection in remote sensing imagery. For example, by integrating a Segment Anything Model (SAM), our approach generates mask-based bounding boxes that serve as the basis for dual sampling: (a) uncertainty estimation to pinpoint challenging samples, and (b) diversity assessment to ensure broad data coverage. Furthermore, our Dynamic Box Switching Module (DBS) addresses the well-known cold start problem for object detection models by replacing its suboptimal initial predictions with SAM-derived masks, thereby enhancing early-stage localization accuracy.

## User Interface

<div style="display: flex;">
  <img src="assets/al_interface.png" alt="AL4FM interface" style="width: 100%;">
</div>

## Repository Layout

The repository is split into two components that are meant to run on different machines and talk to each other over an SSH tunnel:

| Path | Role |
| --- | --- |
| [`backend/`](backend/) | FastAPI server that wraps the object detector, runs SAM-based mask extraction, and serves active learning queries. Runs on a GPU machine. |
| [`frontend/`](frontend/) | PyQt5 annotation desktop tool. Runs on the annotator's workstation and talks to the backend via HTTP. |
| [`backend/object_detectors/`](backend/object_detectors/) | Vendored copies of the object detectors (`rt-detr`, `rf-detr`), the active learning methods, and the preprocessing tools. |
| [`backend/object_detectors/tools/`](backend/object_detectors/tools/) | Offline preprocessing scripts and experiment notebooks (dataset preparation, dense SAM mask extraction, plotting). |

See [`backend/README.md`](backend/README.md) and [`frontend/README.md`](frontend/README.md) for the component-specific documentation.

## Status of this Release

> **Note:** This release reproduces the code that was used for the ICCV 2025 experiments. Two pieces are still being restored from the original lab checkout and are not yet present in the public tree:
>
> - `backend/object_detectors/al_methods/FAST/sample_fast_main.py` — the FAST sampling routine imported by [`backend/server.py`](backend/server.py). Without it, the backend will fail at import time. Smart sampling silently falls back to random sampling if the SAM mask files are missing, but `sample_FAST_images` itself is not yet published.
> - `frontend/utils/startup_dialogs.py` — the startup wizard used by [`frontend/main.py`](frontend/main.py) to collect the server host, port, username, dataset, and object class. Without it the frontend cannot launch.
>
> We are working on restoring both from the original checkout and will push them as soon as they are available. If you need them urgently, please open an issue on GitHub.

## Prerequisites

Backend (server machine):

- Linux with an NVIDIA GPU (>= 8 GB VRAM recommended)
- CUDA 12.x toolchain compatible with your PyTorch build
- >= 32 GB RAM
- Anaconda / Miniconda with Python 3.11

Frontend (annotator workstation):

- Linux, macOS or Windows desktop with a working Qt5 runtime
- Anaconda / Miniconda with Python 3.11
- SSH access to the backend machine

## Installation

Clone the repository:

```bash
git clone https://github.com/mburges-cvl/ICCV_AL4FM.git
cd ICCV_AL4FM
```

### Backend

```bash
cd backend
conda env create -f environment.yml
conda activate al4fm-backend
```

The conda environment installs PyTorch, FastAPI, rasterio and the other scientific dependencies. The `pip:` block of `environment.yml` installs `segment-geospatial` (used as the SAM wrapper), `segment-anything-hq`, and the FastAPI server stack.

If you plan to run the box-prompted SAM preprocessing in [`backend/object_detectors/tools/extract_sam_box_masks.py`](backend/object_detectors/tools/extract_sam_box_masks.py), keep the `autodistill-grounded-sam` entry in `backend/requirements.txt`; otherwise you can drop it.

### Frontend

```bash
cd frontend
conda env create -f environment.yml
conda activate al4fm-frontend
```

The frontend environment only needs PyQt5, Pillow, rasterio, matplotlib and NumPy — it does not require a GPU.

## Dataset Layout

AL4FM expects each dataset to live inside `datasets/<dataset_name>/` and to follow a COCO-style layout:

```
datasets/<dataset_name>/
├── images/                          # 1024×1024 tiles (see note below)
├── annotations.json                 # COCO annotations file (images + categories required)
├── boxes_<dataset_name>_16.npy      # SAM box proposals (produced by preprocessing)
├── img_idx_<dataset_name>_16.npy    # Per-proposal image indices
└── objects_<dataset_name>_16.npy    # Dense SAM masks
```

The three `.npy` files are produced by the SAM preprocessing step (see below) and are consumed by the backend at [`backend/server.py:487-511`](backend/server.py#L487-L511) during smart sampling. The fourth file, `objects_al_gt.npy`, is produced per-experiment in the server work directory and is not part of the dataset.

> **Tile size.** The current pipeline is tuned for 1024×1024 tiles. Datasets sampled at a different size — for example DIOR at 800×800 — are automatically resized and padded when imported through the frontend's dataset wizard, but the aspect ratio has to be square. If you need to support non-square tiles, re-tile the imagery before importing.

For how the main remote sensing benchmarks (DIOR, HRSC2016, DOTAv2, FAIR1M, fMoW, xView) were converted to this layout, see the preprocessing notebooks under [`backend/object_detectors/tools/`](backend/object_detectors/tools/) (`preprocess_*.ipynb`).

## Running the System

The system is designed around three steps: (1) prepare a dataset and extract SAM masks, (2) start the backend, (3) start the frontend and annotate.

### 1. Prepare a Dataset and Extract SAM Masks

You can either let the frontend trigger mask extraction on the server via its dataset wizard, or run the standalone preprocessing script:

```bash
cd backend
python object_detectors/tools/extract_sam_dense_masks.py \
    --dataset <dataset_name> \
    --sam_model_type vit_h \
    --pixels_between_points 16
```

This populates `boxes_*.npy`, `img_idx_*.npy` and `objects_*.npy` under the dataset directory.

### 2. Start the Backend

```bash
cd backend
conda activate al4fm-backend
python server.py --port 8005 --cuda 0 --workdir server_workdir --config <path-to-rtdetr-config>
```

Notable flags (full list in [`backend/server.py`](backend/server.py)):

- `--port` – port the FastAPI server listens on (default: `8005`)
- `--cuda` – CUDA device index
- `--config` – path to the RT-DETRv2 training config to use for the detector
- `--epochs`, `--batch_size`, `--learning_rate`, `--eval_every` – training hyperparameters
- `--workdir` – directory where per-experiment artefacts are written

The server exposes the following HTTP routes to the frontend:

| Route | Method | Purpose |
| --- | --- | --- |
| `/get_next_active_image_ids` | POST | Initial sampling round |
| `/get_new_training_ids_smart` | GET | Active learning round (FAST sampling) |
| `/train_model` | POST | Train the detector on the current labeled set |
| `/evaluate_model` | GET | Evaluate the detector |
| `/extract_prototype_masks` | GET | Run GT-mask extraction |
| `/extract_prototype_features` | GET | Cache detector features |
| `/extract_dense_masks` | POST | Run dense SAM mask extraction on a dataset |
| `/extraction_progress` | GET | Poll progress of a long-running extraction |
| `/get_available_objects`, `/check_dataset_status`, `/prepare_dataset_sync`, `/upload_dataset_file`, `/finalize_dataset_sync` | — | Dataset management endpoints used by the frontend's dataset wizard |

### 3. Open an SSH Tunnel and Start the Frontend

From the annotator's workstation, open an SSH tunnel to the backend port (the tunnel port on the left must match whatever `--port` you started the server with, and must match the port entered in the frontend's startup wizard):

```bash
ssh -N -L 8005:localhost:8005 user@server_address
```

Then start the frontend:

```bash
cd frontend
conda activate al4fm-frontend
python main.py
```

The startup wizard collects the server port, username, dataset, and object class, and then launches the main annotation window. See [`frontend/README.md`](frontend/README.md) for the in-app workflow.

## Method

<div style="display: flex;">
  <img src="assets/active_learning.png" alt="AL4FM method" style="width: 100%;">
</div>

## Results

We validate the performance of AL4FM on the DIOR, HRSC2016, DOTAv2, FAIR1M and our private Wafflehome dataset.

<table style="width: 100%; border-collapse: collapse; border-spacing: 0; margin: 0 auto;">
  <tr>
    <td style="width: 20%; text-align: center; vertical-align: top; padding: 0;">
      <img src="assets/dior_AP50_curve.svg" alt="DIOR" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
      <p style="text-align: center; margin: 5px 0 0;">DIOR</p>
    </td>
    <td style="width: 20%; text-align: center; vertical-align: top; padding: 0;">
      <img src="assets/hrsc2016_AP50_curve.svg" alt="HRSC2016" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
      <p style="text-align: center; margin: 5px 0 0;">HRSC2016</p>
    </td>
    <td style="width: 20%; text-align: center; vertical-align: top; padding: 0;">
      <img src="assets/dotav2_AP50_curve.svg" alt="DOTAv2" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
      <p style="text-align: center; margin: 5px 0 0;">DOTAv2</p>
    </td>
    <td style="width: 20%; text-align: center; vertical-align: top; padding: 0;">
      <img src="assets/fair1m_AP50_curve.svg" alt="FAIR1M" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
      <p style="text-align: center; margin: 5px 0 0;">FAIR1M</p>
    </td>
    <td style="width: 20%; text-align: center; vertical-align: top; padding: 0;">
      <img src="assets/wafflehome_AP50_curve.svg" alt="Wafflehome" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
      <p style="text-align: center; margin: 5px 0 0;">Wafflehome</p>
    </td>
  </tr>
</table>

## License

Both the front- and backend are released under the MIT license. See [LICENSE](LICENSE).

## Acknowledgements

Our work is built upon [Segment Anything](https://github.com/facebookresearch/segment-anything), [SAM Geo](https://samgeo.gishub.org/) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR). This research was supported in part by an appointment to the Oak Ridge National Laboratory GRO Program, sponsored by the U.S. Department of Energy and administered by the Oak Ridge Institute for Science and Education.

## Citation

If you find our work useful for your research, please consider citing our ICCV paper (BibTeX TBD).
