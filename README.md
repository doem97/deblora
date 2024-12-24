Source code for "De-Biased Representation Learning for Long-tailed PEFT in Remote Sensing".

## 🛠️ Installation
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

Execute the following command to run the deblora:

**Feature Extraction and Clustering**:
   ```bash
   bash ./feat_cluster_lora_kmeans.sh
   ```

**Feature Calibration and Linear Probing**:
   ```bash
   # Feature Calibration scripts and pipelines are under preparation
   ```