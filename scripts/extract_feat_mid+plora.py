import sys

sys.path.append("/workspace/dso/gensar/dift")
import argparse
import os
import torch
from PIL import Image
from src.models.dift_sd_lora import SDFeaturizer4Eval
import numpy as np
from tqdm import tqdm


def process_images_in_folder(
    folder_path, dift, category, img_size, time_steps, up_ft_index, ensemble_size, output_path, worker_idx, n_workers, mid_feat
):
    """
    Process and extract features from images in a specified folder.

    Parameters:
    - folder_path: Path to the folder containing images.
    - dift: Initialized SDFeaturizer4Eval object.
    - category: Image category.
    - img_size: Image size for processing.
    - time_steps: List of time steps for feature extraction.
    - up_ft_index: Upsampling block index for feature extraction.
    - ensemble_size: Number of images in each batch for feature extraction.
    - output_path: Path to save extracted features.
    - worker_idx: Index of the current worker.
    - n_workers: Total number of workers.
    - mid_feat: Use mid-level features for extraction.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".bmp"))]
    # Split the files among workers
    worker_files = np.array_split(image_files, n_workers)[worker_idx]
    print(f"Worker {worker_idx} processing {len(worker_files)} images in category '{category}'...")
    images_processed = 0

    # Choose to use tqdm progress bar only for worker 0
    iterable_files = tqdm(worker_files, desc=f"Processing images in '{category}'") if worker_idx == 0 else worker_files
    for img_filename in iterable_files:
        img_path = os.path.join(folder_path, img_filename)
        img = Image.open(img_path).convert("RGB")

        ft = []

        # Choose to use tqdm progress bar for time steps only for worker 0
        iterable_time_steps = tqdm(time_steps, desc="Time steps", leave=False) if worker_idx == 0 else time_steps
        if mid_feat:
            for t in iterable_time_steps:
                # Extract feature
                feature = dift.mid_forward(img, category, img_size, t, ensemble_size)
                ft.append(feature)
        else:
            for t in iterable_time_steps:
                # Extract feature
                feature = dift.forward(img, category, img_size, t, up_ft_index, ensemble_size)
                ft.append(feature)

        # Prepare output filename and save feature tensor
        feature_filename = os.path.splitext(img_filename)[0] + ".pt"
        feature_path = os.path.join(output_path, category, feature_filename)
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        torch.save(torch.stack(ft).squeeze(0).cpu(), feature_path)

        images_processed += 1

    if worker_idx == 0:
        print(f"Processed {images_processed} images in category '{category}'.")


def main(args):
    print(
        f"Runing >>>>>>>>>>>>> Worker {args.worker_idx} of {args.n_workers} workers extracting features from '{args.input_path}' to '{args.output_path}'"
    )
    img_size = [args.img_size, args.img_size]
    if args.sample_step is not None:
        time_steps = [args.sample_step]
    else:
        time_steps = np.linspace(1, 999, args.sample_num, dtype=int).tolist()

    for category in os.listdir(args.input_path):
        dift = SDFeaturizer4Eval(
            sd_id=args.sd_id, cat_list=["cargo", "fishing", "container", "tanker"], prompt_prefix=args.prompt_prefix, plora_path=args.plora_path, category=category
        )
        folder_path = os.path.join(args.input_path, category)
        if os.path.isdir(folder_path):
            process_images_in_folder(
                folder_path,
                dift,
                category,
                img_size,
                time_steps,
                args.up_ft_index,
                args.ensemble_size,
                args.output_path,
                args.worker_idx,
                args.n_workers,
                args.mid_feat,
            )
            print(f"Completed processing for category '{category}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from input images in a folder, and save them as torch tensors.")
    parser.add_argument("--img_size", type=int, default=512, help="Resize input image to this size before processing. Default is 512x512.")
    parser.add_argument("--sd_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Model ID of the diffusion model in Hugging Face.")
    parser.add_argument(
        "--sample_num", type=int, default=10, help="Number of time-steps to sample for diffusion, chosen uniformly from range [0, 1000]."
    )
    parser.add_argument("--sample_step", type=int, default=None, help="Time-step to sample for diffusion.")
    parser.add_argument("--up_ft_index", type=int, default=3, choices=[0, 1, 2, 3], help="Upsampling block index for feature map extraction.")
    parser.add_argument("--mid_feat", action="store_true", help="Use mid-level features for extraction.")
    parser.add_argument("--ensemble_size", type=int, default=8, help="Number of repeated images in each batch for feature extraction.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image folder.")
    parser.add_argument("--output_path", type=str, default="./data/features", help="Path to save the output features as torch tensors.")
    parser.add_argument("--n_workers", type=int, default=1, help="Total number of parallel workers.")
    parser.add_argument("--worker_idx", type=int, default=0, help="The index of the current worker (starting from 0).")
    parser.add_argument("--prompt_prefix", type=str, default=None, help="Prompt prefix for the diffusion model.")
    parser.add_argument("--plora_path", type=str, default=None, help="Path to the LoRA model.")
    args = parser.parse_args()
    main(args)
