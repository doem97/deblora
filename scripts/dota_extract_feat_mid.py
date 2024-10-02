import sys

sys.path.append("/workspace/dso/gensar/dift")
import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models.dift_sd_lora import SDMidFeatExtract

DOTAV1_CAT_LIST = [
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "ground-track-field",
    "harbor",
    "helicopter",
    "large-vehicle",
    "plane",
    "roundabout",
    "ship",
    "small-vehicle",
    "soccer-ball-field",
    "storage-tank",
    "swimming-pool",
    "tennis-court",
]


def main(args):
    print(
        f"Runing >>>>>>>>>>>>> Worker {args.worker_idx} of {args.n_workers} workers extracting features from '{args.input_path}' to '{args.output_path}'"
    )
    assert "category_name" in args.prompt_template, "Prompt template must contain 'category_name' for category substitution."
    dift = SDMidFeatExtract(sd_id=args.sd_id, cat_list=DOTAV1_CAT_LIST, prompt_template=args.prompt_template)
    img_size = [args.img_size, args.img_size]
    # INFO: history version uses a list. It will conflict with folder naming (xxx_t) so just pass a single name everytime.
    time_steps = args.time_steps

    for category in DOTAV1_CAT_LIST:
        for split in ["train", "val"]:
            input_path = os.path.join(args.input_path, split, category)
            output_path = os.path.join(args.output_path, split, category)
            if os.path.isdir(input_path):
                image_files = [f for f in os.listdir(input_path) if f.lower().endswith((".png"))]
                # Split the files among workers
                worker_files = np.array_split(image_files, args.n_workers)[args.worker_idx]
                print(f"Worker {args.worker_idx} processing {len(worker_files)} images in category '{category}'...")

                images_processed = 0

                # Choose to use tqdm progress bar only for worker 0
                iterable_files = tqdm(worker_files, desc=f"Processing '{category}' for {split} set") if args.worker_idx == 0 else worker_files
                for img_filename in iterable_files:
                    img_path = os.path.join(input_path, img_filename)
                    img = Image.open(img_path).convert("RGB")

                    ft = []

                    # Choose to use tqdm progress bar for time steps only for worker 0
                    iterable_time_steps = tqdm(time_steps, desc="Time steps", leave=False) if args.worker_idx == 0 else time_steps
                    for t in iterable_time_steps:
                        # Extract mid feature
                        if args.use_cat_prompt:
                            feature = dift.forward(img, category, img_size, t, args.ensemble_size)
                        else:
                            feature = dift.forward(img, "", img_size, t, args.ensemble_size)
                        ft.append(feature)

                    # Prepare output filename and save feature tensor
                    feature_filename = os.path.splitext(img_filename)[0] + ".pt"
                    feature_path = os.path.join(output_path, feature_filename)
                    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
                    torch.save(torch.stack(ft).squeeze(0).cpu(), feature_path)

                    images_processed += 1

                if args.worker_idx == 0:
                    print(f"Processed {images_processed} images in category '{category}'.")

                print(f"Completed processing for category '{category}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from input images in a folder, and save them as torch tensors.")
    parser.add_argument("--img_size", type=int, default=512, help="Resize input image to this size before processing. Default is 512x512.")
    parser.add_argument("--sd_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Model ID in HuggingFace or path to local folder.")
    parser.add_argument("--time_steps", type=int, nargs="+", default=[1], help="List of time-steps to sample for diffusion. Default is [1].")
    parser.add_argument("--ensemble_size", type=int, default=8, help="Number of repeated images in each batch for feature extraction.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the folder of images for feature extraction.")
    parser.add_argument("--output_path", type=str, default="./data/features", help="Path to save the output features as torch tensors `.pt`.")
    parser.add_argument("--n_workers", type=int, default=1, help="Total number of parallel workers, functioning by split data.")
    parser.add_argument("--worker_idx", type=int, default=0, help="The index of the current worker (starting from 0).")
    parser.add_argument("--prompt_template", type=str, default="aerial image of a category_name", help="Prompt used for the diffusion model.")
    parser.add_argument("--use_cat_prompt", type=bool, default=True, help="Use null prompt during feature extraction instead of category prompts.")

    args = parser.parse_args()
    main(args)
