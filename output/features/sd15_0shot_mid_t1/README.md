Please put the extracted sd15 zero-shot inference features here and train linear probing head on it. We use `deblora/scripts/dota_extract_feat_mid.py` to extract these features. 

For easy re-produce we also provide the ready-to-use features in Hugging Face Hub. The download link is (please install `hf_transfer` first):

```bash
# Install hf_transfer   
pip install hf_transfer

# Enable acceleration download
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download the file
huggingface-cli download doem1997/rs_lt ./0shot_mid_t1.tar.gz --repo-type dataset --local-dir ./

# Unzip the file
tar -xzf 0shot_mid_t1.tar.gz -C ./
```

Put the extracted features in current directory (`deblora/output/features/sd15_0shot_mid_t1/`).