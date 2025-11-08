# AL-Frontend: Active Learning Image Annotation Tool
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">

## Description
AL-Frontend is a Python-based tool designed for interactive annotation of objects in aerial and satellite imagery using active learning techniques. The tool allows users to annotate objects with bounding boxes, and features both random and smart (active learning) annotation modes. The application provides real-time performance metrics (AP50, AP75) as annotations accumulate.

![UI Overview](/images/al_interface.png)


## Features
- Interactive detection and annotation of objects in aerial and satellite imagery
- Two annotation modes: Random and Active Learning (smart)
- Support for positive and negative samples via bounding boxes
- Real-time performance metrics (AP50, AP75) visualization
- Multiple dataset management with dataset synchronization
- Image normalization and enhancement controls
- Support for creating new datasets from TIF files
- User activity logging for performance analysis
- Pan/draw interaction modes
- Confidence filtering for predicted annotations
- Support for various image formats: JPG, PNG, TIFF

## Important Files & Folders

- datasets --> all tifs that are converted to a dataset are stored here or folders that are imported.
- datasets/XX/images --> contains the tiles of XX.tif. NOTE: Only tiles of size 1024x1024 are supported at the moment.
- datasets/XX/annotations.json --> a json in the COCO format containing at least all images with unique IDs and the categories. No annotations required.
- images --> contains images present in the readme.md
- logs --> contains the logs of the frontend.
- utils --> contains additional python code (for the start up wizard and the dataset syncing)
- main.py --> start point of the front end.


## Prerequisites
- Python 3.11 or higher
- Anaconda or Miniconda
- CUDA-compatible GPU (recommended, for the server)
- At least 4 GB of VRAM (8GB recommended, for the server)
- At least 32 GB of RAM (for the server)

## Known Issues

- Only supports a single class for annotation (though the model and active learning method do support multi-class)
- Mask extraction is very slow
- If mask extraction is cancelled before completion, weird state occurs where the server crashes during training due to the lack of masks, but the initial check does not check for masks.
- Currently uses wandb for training which should be removed
- Requires 1024x1024 images, does not work with anything differently and will resize and pad input image folders.
- Currently does not keep a hidden validation set, i.e. it trains on the training set and also validates on the training set, thus our model is very prone to overfitting.
- The dataset and class selection should always be the same (the class currently saves the annotations for that dataset)

## Installation
Clone the repository:
```
git clone https://github.com/mburges-cvl/ICCV_AL4FM/tree/main/frontend
cd alfrontend
```

Create and activate a Conda environment:
```
conda env create -f environment.yml
conda activate alfrontend
```

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Connect to server
```
ssh -N -L 8020:localhost:8020 user@server_address
```

Activate the Conda environment:
```
conda activate alfrontend
```

For local usage, run:
```
python main.py
```

## Workflow
1. Start the application and follow the setup wizard
2. Select or create a dataset
3. Choose an object to annotate
4. Use the draw tool to create positive bounding boxes
6. Save annotations when complete

## Dataset Management
- Use existing datasets from the `datasets` directory
- Create new datasets from TIF files with customizable tile settings
- Synchronize datasets between local and remote servers

## Results Analysis
The tool records user activity for further analysis. Results can be found in:
- `logs` - User activity logs in JSONL format

## License
This project is licensed under the MIT License - see the LICENSE file for details.
