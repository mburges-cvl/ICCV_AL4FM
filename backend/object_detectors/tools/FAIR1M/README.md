---
language: en
license: unknown
task_categories:
- object-detection
paperswithcode_id: FAIR1M
pretty_name: FAIR1M
tags:
- remote-sensing
- earth-observation
- geospatial
- satellite-imagery
- object-detection
---

# FAIR1M

<!-- Dataset thumbnail -->
![FAIR1M](./thumbnail.jpg)

<!-- Provide a quick summary of the dataset. -->
The FAIR1M dataset is a fine-grained object recognition and detection dataset that focuses on high-resolution (0.3-0.8m) RGB images taken by the Gaogen (GF) satellites and extracted from Google Earth. It consists of a collection of 15,000 high-resolution images that cover various objects and scenes. The dataset provides annotations in the form of rotated bounding boxes for objects belonging to 5 main categories (ships, vehicles, airplanes, courts, and roads), further divided into 37 sub-categories.
- **Paper:** https://arxiv.org/abs/2103.05569
- **Homepage:** https://www.gaofen-challenge.com/benchmark

## Description

<!-- Provide a longer summary of what this dataset is. -->

FAIR1M is a part of the ISPRS Benchmark on Object Detection in High-Resolution Satellite Images. Please note that, as of now, only a portion of the training dataset (1,732/15,000 images) has been released for the challenge.

- **1 million object instances**
- **Number of Samples**: 15000
- **Bands**: 3 (RGB)
- **Image Size**: 1024x1024
- **Image Resolution**: 0.3–0.8m
- **Land Cover Classes**: 37
- **Classes**: 5 object categories, 37 object sub-categories.
- **Scene Categories**: Passenger Ship, Motorboat, Fishing Boat, Tugboat, other-ship, Engineering Ship, Liquid Cargo Ship, Dry Cargo Ship, Warship, Small Car, Bus, Cargo Truck, Dump Truck, other-vehicle, Van, Trailer, Tractor, Excavator, Truck Tractor, Boeing737, Boeing747, Boeing777, Boeing787, ARJ21, C919, A220, A321, A330, A350, other-airplane, Baseball Field, Basketball Court, Football Field, Tennis Court, Roundabout, Intersection, Bridge
- **Source**: Gaofen/Google Earth


## Usage

To use this dataset, simply use `datasets.load_dataset("blanchon/FAIR1M")`.
<!-- Provide any additional information on how to use this dataset. -->
```python
from datasets import load_dataset
FAIR1M = load_dataset("blanchon/FAIR1M")
```

## Citation 

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->
If you use the FAIR1M  dataset in your research, please consider citing the following publication:


```bibtex
@article{sun2021fair1m,
    title     = {FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery},
    author    = {Xian Sun and Peijin Wang and Zhiyuan Yan and F. Xu and Ruiping Wang and W. Diao and Jin Chen and Jihao Li and Yingchao Feng and Tao Xu and M. Weinmann and S. Hinz and Cheng Wang and K. Fu},
    journal   = {Isprs Journal of Photogrammetry and Remote Sensing},
    year      = {2021},
    doi       = {10.1016/j.isprsjprs.2021.12.004},
    bibSource = {Semantic Scholar https://www.semanticscholar.org/paper/6d3c2dc63ff0deec10f60e5a515c93af4f8676f2}
}
```
