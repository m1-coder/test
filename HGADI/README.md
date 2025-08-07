# HGADI
This repository contains the source code for the paper titled "Inductive Link Prediction in Heterogeneous
Information Networks via Adversarial Distillation". The current version is provided solely for the purpose of peer review. The full codebase, including preprocessing scripts, model architectures, and evaluation metrics, will be uploaded to this GitHub repository after the paper is officially published. 

### Environment
pytorch 2.2.0

python 3.8.18

pytorch-geometric 2.4.0

cuda 12.1

### Train HGADI
`python HIN_Inductive_teacher.py`

### Datasets
The experiments in this study utilize the HGBDataset from the torch_geometric.datasets module of PyTorch Geometric (PyG). 

You can download the preprocessed train/validation/test split files from the following link:https://drive.google.com/file/d/1NSxSKVCNhrtjRm9Ku7HyqJpzfZaHaShg/view?usp=sharing
After downloading, please unzip the file and place the contents in the ./HGADI/ directory.
### External Data Access
Due to GitHub's file size limit, large dataset files required to run our experiments are hosted on Google Drive.
You can download the necessary data using the following link:
https://drive.google.com/file/d/1RVfYSaPiEle803nk70LKgDQNbONwtxQS/view?usp=sharing
After downloading, please place the file under the ./HGADI/ directory.

### Disclaimer
Note: The findings and methodologies presented are part of ongoing research and have not yet been formally published. The authors reserve all rights to the content until official publication.
