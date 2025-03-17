Please download and setup the DOTA_v2 recognition dataset from the following link:

```
# Install hf_transfer dependency (for large file transfer)
pip install hf_transfer

# Enable acceleration download
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download dataset
huggingface-cli download doem1997/rs_lt ./dota.tar --repo-type dataset --local-dir ./

# Unzip to current directory
tar -xf dota.tar -C ./
```

For remaining datasets please download following commands in `deblora/data/hf_dataset.sh`.