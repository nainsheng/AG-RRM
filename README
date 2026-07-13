# AG-RRM: Angle-Guided Region Response Modeling for Cross-Batch Cattle Face Re-Identification
This repository provides the implementation of the proposed **Angle-Guided Region Response Modeling (AG-RRM)** framework for cross-batch cattle face re-identification.

AG-RRM introduces a pose-aware identity representation by modeling the relationship between cattle head yaw and semantic facial region responses. The framework integrates global identity features, fine-grained part features, and pose-dependent region responses to improve cattle identity matching under cross-batch conditions.


## Framework Overview

<p align="center">
  <img src="fig/framework.png" width="95%">
</p>

<p align="center">
  <em>Overall architecture of the proposed AG-RRM framework.</em>
</p>


## Dataset

The experiments are conducted on the proposed **CrossBatch-Simmental** dataset for cross-batch cattle face re-identification.

The dataset contains:

- **22,572** cattle face images
- **286** Simmental cattle identities
- Two acquisition batches with approximately **three months interval**


The dataset is divided into three subsets:

| Split | Images | Identities |
|------|-------:|-----------:|
| Train | 14,458 | 204 |
| Gallery | 4,001 | 82 |
| Probe | 4,113 | 82 |


The dataset can be downloaded from the following links:

### Train Set
[Google Drive] https://drive.google.com/file/d/1E4CqbtYqD9QG6kj3V8kDhuib3B8dsFDR/view?usp=sharing

### Gallery Set
[Google Drive] https://drive.google.com/file/d/13kYAL8CIYQdQjZnUR2PzndTYAeDCWXd-/view?usp=sharing

### Probe Set
[Google Drive] https://drive.google.com/file/d/1Xl5lILA_vllvAWmG1PqcbY4Tq7x3rhqj/view?usp=sharing

### Pretrained Model
[Google Drive] https://drive.google.com/file/d/11aWrzhWoYfc9HiJveAuF2g3w5sUC6YUS/view?usp=sharing


After downloading, place the model file as:

```
model/
└── best_model.pth
```


## Code Structure

```
AG-RRM/
│
├── README.md
├── requirements.txt
│
├── eval.py
├── EfficientNetB3_AGRRM_ReID.py
├── FineGrainedDataset.py
│
├── model/
│   └── best_model.pth
│
└── dataset/
    ├── train/
    │   ├── image/
    │   ├── angle/
    │   └── json/
    │
    ├── gallery/
    │   ├── image/
    │   ├── angle/
    │   └── json/
    │
    └── probe/
        ├── image/
        ├── angle/
        └── json/
```


## Data Format

Each dataset split contains three folders:

```
split/
├── image/
├── angle/
└── json/
```


where:

- `image/` contains cattle face images;
- `angle/` contains cattle head yaw angle annotations;
- `json/` contains face bounding-box annotations.


During evaluation, the corresponding face bounding box is used to crop the cattle face region. The cropped images are resized to \(448 \times 448\) and normalized before feature extraction.


## Evaluation Code

The evaluation code is provided in:

```
eval.py
```

The script extracts feature embeddings from gallery and probe images and evaluates the retrieval performance using:
- Top-1 accuracy
- Top-5 accuracy
- Top-10 accuracy
- mean Average Precision (mAP)


## Requirements
The code is implemented with PyTorch.
Required packages:

```
torch
torchvision
numpy
Pillow
tqdm
```


## Citation

If you use this dataset or code in your research, please cite the corresponding paper:

```
Citation information will be added after publication.
```


## License
The dataset and code are released for academic research purposes only.