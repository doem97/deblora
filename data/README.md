Please download datasets DOTAv1 and FUSRS and symbol link to `deblora/data` folder like:
```
ln -s /path/to/datasets/dotav1 ./datasets/dotav1
ln -s /path/to/datasets/fusrs ./datasets/fusrs
```

**Download links:** [Google Drive](https://drive.google.com/drive/folders/19qGGf4uEfNZmi5wIMzfXMYJkPtyXmyoo?usp=sharing)

### Usage
The datasets should be extracted into `deblora/data` folder by:

```bash
cp dosrs_v1.tar.gz fusrs_v2.tar.gz datasets/
tar -xvzf dosrs_v1.tar.gz
tar -xvzf fusrs_v2.tar.gz
```

The final folder structure should be like:
```
|-- data # our proposed datasets
|   |-- dosrs_v1 # ORS ship dataset
|   |-- fusrs_v2 # SAR ship dataset
|   `-- README.md
|-- exp
|-- ...
`-- output
```

### DOSRS-v1.0
Please refer to [DOSRS-v1.0](dosrs/README.md) for more details.

### FUSRS-v2.0
Please refer to [FUSRS-v2.0](fusrs/README.md) for more details.