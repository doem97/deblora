# debLoRA: Learning De-Biased Representations for Remote-Sensing Imagery

**NeurIPS 2024** &ensp;|&ensp; **Zichen Tian**, **Zhaozheng Chen**, **Qianru Sun** &ensp;|&ensp; *Singapore Management University*

<p>
  <a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/6a8e10164a90d5c3660c3949289f969a-Abstract-Conference.html">
    <img src="https://img.shields.io/badge/📄%20Paper-NeurIPS%202024-8c1b13?style=flat-square" alt="Paper">
  </a>
  <a href="https://arxiv.org/abs/2410.04546">
    <img src="https://img.shields.io/badge/📝%20ArXiv-2410.04546-b31b1b?style=flat-square" alt="ArXiv">
  </a>
</p>

Official implementation of **debLoRA** from the paper **"Learning De-Biased Representations for Remote-Sensing Imagery"** (NeurIPS 2024).


## 🛠️ Installation & Setup
Clone the repository
```bash
git clone https://github.com/doem97/deblora.git
```

### Dataset Setup

The folder structure should be like:
```
./
|-- data
|-- exp
|-- output
|-- scripts
|-- src
|-- .gitignore
`-- README.md
```

Please download datasets with [huggingface-cli script](./output/features/hf_features.sh) and symbol link to `./data` folder. See [Datasets Documentation](data/README.md) for more details.

### Environment Setup

To set up the deblora environment, follow these steps:

1. Create a new conda environment (recommended):
   ```
   conda create -n deblora python=3.8
   conda activate deblora
   ```

2. Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

## 🚀 Usage

**Feature extraction for 0 Shot, Fine-tuned, and LoRA**:

```bash
# feature extraction for 0 shot and fine-tuned
bash ./exp/extract_0shot_feat.sh
bash ./exp/extract_finetune_feat.sh
# feature extraction for LoRA (feature calibration source for pLoRA)
bash ./exp/extract_lora_feat.sh
```

> For easy re-produce, we also provided the ready-to-use extracted features (download links in [hf_features.sh](./output/features/hf_features.sh)). You could directly download the 0shot/fine-tuned/LoRA/pLoRA features by executing the script.

**Linear probing for 0 Shot**:

```bash
bash ./exp/0shot_linprob.sh
```

**Linear probing for Fine-tuned**:

```bash
bash ./exp/ft_linprob.sh
```

**Linear probing for LoRA**:

```bash
bash ./exp/lora_linprob.sh
```

**Feature clustering and calibration for pLoRA**:

```bash
bash ./exp/feat_cluster_lora_kmeans.sh
```

**Linear probing for pLoRA**:

```bash
bash ./exp/plora_linprob.sh
```

Please note that the current code may be poorly organized and is in separate modules. The remaining CLIP code and data will be released in our upcoming work Meta-LoRA (https://github.com/doem97/metalora) with more streamlined pipelines. Please do not hesitate to raise repo issues if you have problems.

## ✍️ Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{NEURIPS2024_6a8e1016,
 author = {Tian, Zichen and Chen, Zhaozheng and Sun, Qianru},
 booktitle = {Advances in Neural Information Processing Systems},
 doi = {10.52202/079017-1848},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {57970--57992},
 publisher = {Curran Associates, Inc.},
 title = {Learning De-Biased Representations for Remote-Sensing Imagery},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/6a8e10164a90d5c3660c3949289f969a-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
