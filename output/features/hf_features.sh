#!/bin/bash
# Author: Doem1997
# Note: requires `pip install hf_transfer`

################################################################################
# Upload - used by @doem1997 only
################################################################################
# split large file into 20GB parts
# split -b 21474836480 -d sd15_0shot_mid_t1.tar.gz "sd15_0shot_mid_t1.tar.gz.part"
# split -b 21474836480 -d sd15+ft_mid_t1.tar.gz "sd15+ft_mid_t1.tar.gz.part"
# split -b 21474836480 -d sd15+lora_mid_t1.tar.gz "sd15+lora_mid_t1.tar.gz.part"

# # Run with huggingface-cli version 0.25.0
# export HF_HUB_ENABLE_HF_TRANSFER=1
# # huggingface-cli login
# huggingface-cli upload doem1997/deblora ./sd15_0shot_mid_t1.tar.gz.part00 --repo-type dataset
# huggingface-cli upload doem1997/deblora ./sd15_0shot_mid_t1.tar.gz.part01 --repo-type dataset

# huggingface-cli upload doem1997/deblora ./sd15+ft_mid_t1.tar.gz.part00 --repo-type dataset
# huggingface-cli upload doem1997/deblora ./sd15+ft_mid_t1.tar.gz.part01 --repo-type dataset

# huggingface-cli upload doem1997/deblora ./sd15+lora_mid_t1.tar.gz.part00 --repo-type dataset
# huggingface-cli upload doem1997/deblora ./sd15+lora_mid_t1.tar.gz.part01 --repo-type dataset

################################################################################
# Download and Extract - used by users
################################################################################
# Create directories
mkdir -p ./{sd15_0shot_mid_t1,sd15+ft_mid_t1,sd15+lora_mid_t1}

# Download commands
export HF_HUB_ENABLE_HF_TRANSFER=1
# huggingface-cli login
huggingface-cli download doem1997/deblora ./sd15_0shot_mid_t1.tar.gz.part00 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/deblora ./sd15_0shot_mid_t1.tar.gz.part01 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/deblora ./sd15+ft_mid_t1.tar.gz.part00 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/deblora ./sd15+ft_mid_t1.tar.gz.part01 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/deblora ./sd15+lora_mid_t1.tar.gz.part00 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/deblora ./sd15+lora_mid_t1.tar.gz.part01 --repo-type dataset --local-dir ./

# Combine split files
cat sd15_0shot_mid_t1.tar.gz.part* >sd15_0shot_mid_t1.tar.gz
cat sd15+ft_mid_t1.tar.gz.part* >sd15+ft_mid_t1.tar.gz
cat sd15+lora_mid_t1.tar.gz.part* >sd15+lora_mid_t1.tar.gz

# For each archive, first check structure then move
tar -xzf sd15_0shot_mid_t1.tar.gz -C ./sd15_0shot_mid_t1
tar -xzf sd15+ft_mid_t1.tar.gz -C ./sd15+ft_mid_t1
tar -xzf sd15+lora_mid_t1.tar.gz -C ./sd15+lora_mid_t1
