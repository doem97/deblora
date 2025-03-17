Source code for "De-Biased Representation Learning for Long-tailed PEFT in Remote Sensing".

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

Please download datasets DOTAv1 and FUSRS following [Datasets Documentation](data/README.md) and symbol link to `./data` folder like:
```
ln -s /path/to/datasets/dotav1 ./data/dotav1
ln -s /path/to/datasets/fusrs ./data/fusrs
```

See [Datasets Documentation](datasets/README.md) for more details.

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
python ./scripts/dota_extract_feat_mid.py
# feature extraction for LoRA (feature calibration source for pLoRA)
python ./scripts/dota_extract_feat_mid+lora.py
```

> For easy re-produce, we also provided the ready-to-use extracted features (download links in `./output/features/hf_features.sh`). You could directly download the 0shot/fine-tuned/LoRA/pLoRA features by executing the script.

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
bash ./feat_cluster_lora_kmeans.sh
```

**Linear probing for pLoRA**:

```bash
bash ./exp/plora_linprob.sh
```