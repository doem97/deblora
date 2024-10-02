import gc
import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from torchvision.transforms import PILToTensor


def print_blocks(blocks):
    for name, module in blocks.named_children():
        if isinstance(module, nn.ModuleList):
            for sub_module in module:
                print(sub_module.__class__.__name__)
        else:
            print(module.__class__.__name__)


class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        use_mid_features: bool = False,
    ):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            # print("Downsample block")
            # print_blocks(downsample_block)
            # print(sample.shape)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            # print("Mid block")
            # print_blocks(self.mid_block)
            # print(sample.shape)
        if use_mid_features:
            mid_ft = sample.detach()
            return mid_ft

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):

            if i > np.max(up_ft_indices):
                break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

            # print("Upsample block")
            # print_blocks(upsample_block)
            # print(sample.shape)

            if i in up_ft_indices:
                up_ft[i] = sample.detach()
        return up_ft


class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_up_ft = self.unet(
            latents_noisy,
            t,
            up_ft_indices,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        return unet_up_ft

    @torch.no_grad()
    def mid_forward(
        self,
        img_tensor,
        t,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_mid_ft = self.unet(
            latents_noisy,
            t,
            up_ft_indices=[],
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            use_mid_features=True,
        )
        return unet_mid_ft


class SDFeaturizer(nn.Module):
    def __init__(self, sd_id="stabilityai/stable-diffusion-1-5", null_prompt=""):
        super().__init__()
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        # INFO: For diffusers v0.26.0, the return of `pipe.encode_prompt` is a tuple of {positive, negative} prompt embeddings.
        null_prompt_embeds, _ = onestep_pipe.encode_prompt(
            prompt=null_prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )  # [1, 77, dim]

        self.null_prompt_embeds = null_prompt_embeds
        self.null_prompt = null_prompt
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self, img_tensor, prompt="", t=261, up_ft_index=1, ensemble_size=8):
        """
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        """
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda()  # ensem, c, h, w
        if prompt == self.null_prompt:
            prompt_embeds = self.null_prompt_embeds
        else:
            prompt_embeds = self.pipe._encode_prompt(
                prompt=prompt,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )  # [1, 77, dim]
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
        )
        unet_ft = unet_ft_all["up_ft"][up_ft_index]  # ensem, c, h, w
        unet_ft = unet_ft.mean(0, keepdim=True)  # 1,c,h,w
        return unet_ft


class SDFeaturizer4Eval(SDFeaturizer):
    def __init__(
        self,
        sd_id="runwayml/stable-diffusion-v1-5",
        null_prompt="",
        cat_list=[],
        prompt_prefix=None,
        lora_path=None,
        clora_path=None,
        plora_path=None,
        category=None,
        checkpoint=1000,
    ):
        super().__init__(sd_id, null_prompt)
        with torch.no_grad():
            cat2prompt_embeds = {}
            for cat in cat_list:
                if prompt_prefix is None:
                    prompt = ""
                else:
                    prompt = f"{prompt_prefix} {cat}"
                prompt_embeds, neg_prompt_embds = self.pipe.encode_prompt(
                    prompt=prompt,
                    device="cuda",
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )  # [1, 77, dim]
                cat2prompt_embeds[cat] = prompt_embeds
            self.cat2prompt_embeds = cat2prompt_embeds

        # self.pipe.tokenizer = None
        # self.pipe.text_encoder = None

        assert [lora_path, plora_path, clora_path].count(
            None
        ) >= 2, "Please make sure you didn't give more than 1 c/raw/pLoRA path to load LORA weights."

        if lora_path is not None:
            print("Loading single LoRA weights...")
            self.pipe.load_lora_weights(lora_path, adaptor_name="ORS", weight_name="pytorch_lora_weights.safetensors")
        elif clora_path is not None:
            assert category is not None, "Please provide the category for the cLORA model."
            _cat_lora_path = os.path.join(clora_path, category, f"checkpoint-{checkpoint}")
            print(f"Loading cLORA weights at {_cat_lora_path}...")
            # self.pipe.load_lora_weights(_cat_lora_path, adaptor_name=f"{category}", weight_name="pytorch_model.bin")
            self.pipe.load_lora_weights(_cat_lora_path, adaptor_name=f"{category}")
        elif plora_path is not None:
            assert category is not None, "Please provide the category for the pLORA model."
            ################ wclora (not working) ##################
            # print("Loading pLoRA weights...")
            # self.pipe.load_lora_weights(os.path.join(plora_path, "cargo"), adaptor_name="cargo", weight_name="pytorch_lora_weights.safetensors")
            # self.pipe.load_lora_weights(os.path.join(plora_path, "container"), adaptor_name="container", weight_name="pytorch_lora_weights.safetensors")
            # self.pipe.load_lora_weights(os.path.join(plora_path, "fishing"), adaptor_name="fishing", weight_name="pytorch_lora_weights.safetensors")
            # self.pipe.load_lora_weights(os.path.join(plora_path, "tanker"), adaptor_name="tanker", weight_name="pytorch_lora_weights.safetensors")
            # self.pipe.load_lora_weights("/workspace/dso/gensar/lora/output/ors_clora/512_fp32_s1600+200ep_wp160_bs32_lr1e-03_rank8/ORS", adaptor_name="ORS", weight_name="pytorch_lora_weights.safetensors")
            # if category == "cargo":
            #     self.pipe.set_adapters("cargo")
            # elif category == "tanker":
            #     self.pipe.set_adapters("tanker")
            # elif category == "container":
            #     self.pipe.set_adapters(["container", "ORS"], adapter_weights=[0.8, 0.2])
            # elif category == "fishing":
            #     self.pipe.set_adapters(["fishing", "ORS"], adapter_weights=[0.9, 0.1])

            ################ plora ##################
            # INFO: adaptor_name is only used in set_adapters, but not in the prompts.
            print("Loading pLoRA weights...")
            self.pipe.load_lora_weights(
                "/workspace/dso/gensar/lora/output/ors_clora/512_fp32_s1600+200ep_wp160_bs32_lr1e-03_rank8/cargo",
                adaptor_name="cargo",
                weight_name="pytorch_lora_weights.safetensors",
            )
            self.pipe.load_lora_weights(
                "/workspace/dso/gensar/lora/output/ors_clora/512_fp32_s1600+200ep_wp160_bs32_lr1e-03_rank8/tanker",
                adaptor_name="tanker",
                weight_name="pytorch_lora_weights.safetensors",
            )
            self.pipe.load_lora_weights(
                os.path.join(plora_path, "512_fp32_s5000_wp160_bs32_lr1e-03_rank8_prototype_0"),
                adaptor_name="prototype0",
                weight_name="pytorch_lora_weights.safetensors",
            )
            self.pipe.load_lora_weights(
                os.path.join(plora_path, "512_fp32_s5000_wp160_bs32_lr1e-03_rank8_prototype_1"),
                adaptor_name="prototype1",
                weight_name="pytorch_lora_weights.safetensors",
            )
            self.pipe.load_lora_weights(
                os.path.join(plora_path, "512_fp32_s5000_wp160_bs32_lr1e-03_rank8_prototype_2"),
                adaptor_name="prototype2",
                weight_name="pytorch_lora_weights.safetensors",
            )
            self.pipe.load_lora_weights(
                os.path.join(plora_path, "512_fp32_s5000_wp160_bs32_lr1e-03_rank8_prototype_3"),
                adaptor_name="prototype3",
                weight_name="pytorch_lora_weights.safetensors",
            )
            if category == "cargo":
                self.pipe.set_adapters("cargo")
            elif category == "tanker":
                self.pipe.set_adapters("tanker")
            elif category == "container":
                self.pipe.set_adapters(["0", "1", "2", "3"], adapter_weights=[0.12, 0.19, 0.37, 0.32])
            elif category == "fishing":
                self.pipe.set_adapters(["0", "1", "2", "3"], adapter_weights=[0.72, 0.08, 0.12, 0.08])

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(
        self,
        img,
        category=None,
        img_size=[512, 512],
        t=261,
        up_ft_index=1,
        ensemble_size=8,
    ):
        if img_size is not None:
            img = img.resize(img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1, 1).cuda()  # ensem, c, h, w
        if category in self.cat2prompt_embeds:
            prompt_embeds = self.cat2prompt_embeds[category]
        else:
            prompt_embeds = self.null_prompt_embeds
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1).cuda()
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
        )
        unet_ft = unet_ft_all["up_ft"][up_ft_index]  # ensem, c, h, w
        unet_ft = unet_ft.mean(0, keepdim=True)  # 1,c,h,w
        return unet_ft

    @torch.no_grad()
    def mid_forward(
        self,
        img,
        category=None,
        img_size=[512, 512],
        t=261,
        ensemble_size=8,
    ):
        if img_size is not None:
            img = img.resize(img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1, 1).cuda()  # ensem, c, h, w
        if category in self.cat2prompt_embeds:
            prompt_embeds = self.cat2prompt_embeds[category]
        else:
            prompt_embeds = self.null_prompt_embeds
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1).cuda()
        unet_mid_ft = self.pipe.mid_forward(
            img_tensor=img_tensor,
            t=t,
            prompt_embeds=prompt_embeds,
        )
        unet_ft = unet_mid_ft["mid_ft"]  # 1280, 8, 8
        unet_ft = unet_ft.mean(0, keepdim=True)  # 1,c,h,w
        return unet_ft


class SDMidFeatExtract(SDFeaturizer):
    """Extract features from images using the middle layer of U-Net of Stable Diffusion model."""

    def __init__(
        self,
        sd_id="runwayml/stable-diffusion-v1-5",
        null_prompt="",
        cat_list=[],
        prompt_template="a photo of a category_name",
    ):
        super().__init__(sd_id, null_prompt)
        assert "category_name" in prompt_template, "Prompt template must contain 'category_name' for category substitution."
        with torch.no_grad():
            cat2prompt_embeds = {}
            for cat in cat_list:
                prompt = prompt_template.replace("category_name", cat)
                # Return: prompt embeddings, negative prompt embeddings.
                prompt_embeds, _ = self.pipe.encode_prompt(
                    prompt=prompt,
                    device="cuda",
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                cat2prompt_embeds[cat] = prompt_embeds
            self.cat2prompt_embeds = cat2prompt_embeds

        # Reduce GPU occupation: the prompt embeddings should be fixed.
        # tokenizer basically is to tokenize the prompt, and text_encoder is to embedding the tokens.
        self.pipe.tokenizer = None
        self.pipe.text_encoder = None

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, img, category=None, img_size=[512, 512], t=1, ensemble_size=8):
        if img_size is not None:
            img = img.resize(img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0).repeat(ensemble_size, 1, 1, 1).cuda()  # ensem, c, h, w
        if category in self.cat2prompt_embeds:
            prompt_embeds = self.cat2prompt_embeds[category]
        else:
            prompt_embeds = self.null_prompt_embeds
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1).cuda()
        # unet_mid_ft: 1280, 8, 8
        unet_mid_ft = self.pipe.mid_forward(
            img_tensor=img_tensor,
            t=t,
            prompt_embeds=prompt_embeds,
        )
        unet_mid_ft_ensem = unet_mid_ft.mean(0, keepdim=True)  # 1,c,h,w
        return unet_mid_ft_ensem


class SDMidFeatExtractLoRA(SDMidFeatExtract):
    """Extract features from images using the middle layer of U-Net of Stable Diffusion model."""

    def __init__(
        self,
        sd_id="runwayml/stable-diffusion-v1-5",
        null_prompt="",
        cat_list=[],
        prompt_template="a photo of a category_name",
        lora_path=None,
    ):
        super().__init__(sd_id, null_prompt, cat_list, prompt_template)
        assert lora_path is not None, "Please provide the path to load LoRA weights."
        print("Loading single LoRA weights...")
        self.pipe.load_lora_weights(lora_path, adaptor_name="ORS", weight_name="pytorch_lora_weights.safetensors")
