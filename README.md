# Active Learning Meets Foundation Models (AL4FM)
This is the repository accompanying the "Active Learning Meets Foundation Models: Fast Remote Sensing Data Annotation for Object Detection" paper accepted at ICCV 2025, the code is released under the MIT license.


Active Learning Meets Foundation Models (AL4FM) is a real-time active learning and semi-automated labeling framework that leverages foundation models to streamline dataset annotation for object detection in remote sensing imagery. For example, by integrating a Segment Anything Model (SAM), our approach generates mask-based bounding boxes that serve as the basis for dual sampling: (a) uncertainty estimation to pinpoint challenging samples, and (b) diversity assessment to ensure broad data coverage. Furthermore, our Dynamic Box Switching Module (DBS) addresses the well-known cold start problem for object detection models by replacing its suboptimal initial predictions with SAM-derived masks, thereby enhancing early-stage localization accuracy.

## User Interface

<div style="display: flex;">
  <img src="assets/al_interface.png" alt="Alt text" style="width: 100%;">
</div>

## News

## Installation

Install the conda requirements file in a [**Python>=3.11**](https://www.python.org/) environment.

```bash
conda install TBD
```

## Method

<div style="display: flex;">
  <img src="assets/active_learning.png" alt="Alt text" style="width: 100%;">
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

Both the front and backend are released under the MIT licence.

## Acknowledgements

Our work is built upon [Segment Anything](https://github.com/facebookresearch/segment-anything), [SAM Geo](https://samgeo.gishub.org/) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR).

## Citation

If you find our work useful for you research, please consider citing our upcoming ICCV paper. (TBD)
