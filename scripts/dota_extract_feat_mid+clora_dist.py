import sys

sys.path.append("/workspace/dso/gensar/dift")
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import PILToTensor
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


class ImageDataset(Dataset):
    def __init__(self, image_files, input_path, img_size):
        self.image_files = image_files
        self.input_path = input_path
        self.img_size = img_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.input_path, img_filename)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        return img_tensor, img_filename


def collate_fn(batch):
    img_tensors, img_filenames = zip(*batch)
    return torch.stack(img_tensors), list(img_filenames)


class SDMidFeatExtractClassLoRA(SDMidFeatExtract):
    """Extract features from images using the middle layer of U-Net of Stable Diffusion model."""

    def __init__(
        self,
        clora_path,
        sd_id="runwayml/stable-diffusion-v1-5",
        null_prompt="",
        cat_list=[],
        prompt_template="a photo of a category_name",
        checkpoint=29,
    ):
        cat_list = DOTAV1_CAT_LIST
        super().__init__(sd_id, null_prompt, cat_list, prompt_template)
        for _cat in cat_list:
            _cat_lora_path = os.path.join(clora_path, _cat, f"checkpoint-{checkpoint}")
            print(f"Loading category LORA weights of {_cat} at {_cat_lora_path}...")
            self.pipe.load_lora_weights(_cat_lora_path, adaptor_name=f"cLoRA_{_cat}", weight_name="pytorch_model.bin")

    @torch.no_grad()
    def forward(self, img_tensors, category=None, t=15, ensemble_size=4):
        img_tensors = img_tensors.repeat_interleave(ensemble_size, dim=0).cuda()  # batch*ensem, c, h, w
        if category in self.cat2prompt_embeds:
            prompt_embeds = self.cat2prompt_embeds[category]
        elif category == self.null_prompt:
            prompt_embeds = self.null_prompt_embeds
        else:
            raise ValueError(f"Category prompt '{category}' not supported.")
        prompt_embeds = prompt_embeds.repeat(img_tensors.size(0), 1, 1).cuda()
        # unet_mid_ft: batch*ensem, c, h, w
        unet_mid_ft = self.pipe.mid_forward(
            img_tensor=img_tensors,
            t=t,
            prompt_embeds=prompt_embeds,
        )
        unet_mid_ft = unet_mid_ft.view(img_tensors.size(0) // ensemble_size, ensemble_size, *unet_mid_ft.shape[1:])  # batch, ensem, c, h, w
        unet_mid_ft_ensem = unet_mid_ft.mean(1)  # batch, c, h, w
        return unet_mid_ft_ensem


def main(args):
    # Initialize the distributed environment
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[INFO] Rank {rank} of {world_size} workers.")
    assert "category_name" in args.prompt_template, "Prompt template must contain 'category_name' for category substitution."
    dift = SDMidFeatExtractClassLoRA(args.clora_path, sd_id=args.sd_id, cat_list=DOTAV1_CAT_LIST, prompt_template=args.prompt_template)
    dift = DDP(dift, device_ids=[rank])
    img_size = [args.img_size, args.img_size]
    time_steps = args.time_steps

    for category in DOTAV1_CAT_LIST:
        for split in ["train", "val"]:
            input_path = os.path.join(args.input_path, split, category)
            output_path = os.path.join(args.output_path, split, category)
            if os.path.isdir(input_path):
                image_files = [f for f in os.listdir(input_path) if f.lower().endswith((".png"))]
                # Split the files among workers
                worker_files = np.array_split(image_files, world_size)[rank]
                print(f"[RUN] Rank {rank} processing {len(worker_files)} images in category '{category}'...")

                dataset = ImageDataset(worker_files, input_path, img_size)
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, sampler=sampler)

                images_processed = 0

                # Choose to use tqdm progress bar only for rank 0
                iterable_dataloader = tqdm(dataloader, desc=f"Processing '{category}' for {split} set") if rank == 0 else dataloader
                for batch_images, batch_files in iterable_dataloader:
                    feat_batch = []

                    # Choose to use tqdm progress bar for time steps only for rank 0
                    iterable_time_steps = tqdm(time_steps, desc="Time steps", leave=False) if rank == 0 else time_steps
                    for t in iterable_time_steps:
                        # Extract mid feature
                        if args.use_cat_prompt:
                            features = dift(batch_images, category, t, args.ensemble_size)
                        else:
                            features = dift(batch_images, "", t, args.ensemble_size)
                        feat_batch.append(features)

                    feat_batch = torch.stack(feat_batch).squeeze(0).cpu()

                    for i, img_filename in enumerate(batch_files):
                        # Prepare output filename and save feature tensor
                        feature_filename = os.path.splitext(img_filename)[0] + ".pt"
                        feature_path = os.path.join(output_path, feature_filename)
                        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
                        torch.save(feat_batch[:, i], feature_path)

                    images_processed += len(batch_files)

                if rank == 0:
                    print(f"Processed {images_processed} images in category '{category}'.")

                print(f"Rank {rank} completed processing for category '{category}'.")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from input images in a folder, and save them as torch tensors.")
    parser.add_argument("--img_size", type=int, default=512, help="Resize input image to this size before processing. Default is 512x512.")
    parser.add_argument("--sd_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Model ID of the diffusion model in Hugging Face.")
    parser.add_argument("--time_steps", type=int, nargs="+", default=[1], help="List of time-steps to sample for diffusion. Default is [1].")
    parser.add_argument("--ensemble_size", type=int, default=8, help="Number of repeated images in each batch for feature extraction.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the folder of images for feature extraction.")
    parser.add_argument("--clora_path", type=str, required=True, help="Path to the LoRA model for feature extraction.")
    parser.add_argument("--output_path", type=str, default="./data/features", help="Path to save the output features as torch tensors `.pt`.")
    parser.add_argument("--n_workers", type=int, default=1, help="Total number of parallel workers, functioning by split data.")
    parser.add_argument("--worker_idx", type=int, default=0, help="The index of the current worker (starting from 0).")
    parser.add_argument("--prompt_template", type=str, default="aerial image of a category_name", help="Prompt used for the diffusion model.")
    parser.add_argument("--use_cat_prompt", type=bool, default=True, help="Use null prompt during feature extraction instead of category prompts.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to process in each batch.")

    args = parser.parse_args()
    main(args)
