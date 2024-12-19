import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.controlnet import MultiControlNetModel
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from modeling.builder import build_projector
class FSDPMixin:
    def fsdp_ignored_modules(self) -> list:
        return []
from modeling.import_utils import is_xformers_available
from modeling.logger import logger
from modeling.utils import randn_tensor, check_path_and_file, get_model_device, get_model_dtype

PluginType = Literal["embedding", "head"]
PipelineImageType = (
    PIL.Image.Image | np.ndarray | torch.FloatTensor | list[PIL.Image.Image] | list[torch.FloatTensor] | list[np.ndarray]
)


class PluginBase(ABC, nn.Module, FSDPMixin):
    initializer_range: float = 0.02
    plugin_type: PluginType | None = None

    def _init_weights(self, module):
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=std)

    @property
    def device(self):
        return get_model_device(self)

    @property
    def dtype(self):
        return get_model_dtype(self)

    @property
    @abstractmethod
    def processor(self):
        """
        In `MultimodalEmbedding`, it is similar to text tokenizer, it processes signals into a form that can be understood by `MultimodalEmbedding`.
        In `MultimodalHead`, it will process signals into a form that can be **trained** by `MultimodalHead`.
        """
        pass

    @property
    @abstractmethod
    def config(self):
        pass

    @abstractmethod
    def save_model(self, output_dir: str):
        pass

    @abstractmethod
    def load_model(self, output_dir: str):
        pass

    @abstractmethod
    def forward(self):
        pass


class MultimodalEmbedding(PluginBase):
    initializer_range: float = 0.02
    plugin_type: PluginType | None = "embedding"

    @property
    @abstractmethod
    def embed_len(self):
        """
        The length a signal after being processed by MultimodalEmbedding.
        """
        pass

    @property
    @abstractmethod
    def embed_dim(self):
        """
        The dimension of the embedding.
        """
        pass


class MultimodalHead(PluginBase):
    initializer_range: float = 0.02
    plugin_type: PluginType | None = "head"

    @abstractmethod
    @torch.no_grad()
    def pipeline(self):
        pass


# embedding modules
class IMAGEEmbedding(MultimodalEmbedding):
    def __init__(
        self,
        pretrained_model_name_or_path: str | None = None,
        num_IMAGE_queries: int = 64,
        embed_hidden_size: int = 4096,
        freeze_IMAGE_queries: bool = False,
    ):
        super().__init__()
        self.save_model_name = "IMAGE_embedding"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.num_IMAGE_queries = num_IMAGE_queries
        self.embed_hidden_size = embed_hidden_size
        self.freeze_IMAGE_queries = freeze_IMAGE_queries

        self.IMAGE_queries = nn.Parameter(torch.zeros(1, self.num_IMAGE_queries, self.embed_hidden_size))
        self._init_weights(self.IMAGE_queries)

        if pretrained_model_name_or_path is not None:
            self.load_model(pretrained_model_name_or_path)

        self.IMAGE_queries.requires_grad_(not freeze_IMAGE_queries)

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        if self.freeze_IMAGE_queries:
            ignored_modules.append(self)
        return ignored_modules

    @property
    def processor(self):
        return None

    @property
    def embed_len(self):
        return self.num_IMAGE_queries

    @property
    def embed_dim(self):
        return self.embed_hidden_size

    @property
    def config(self):
        return dict(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            num_IMAGE_queries=self.num_IMAGE_queries,
            embed_len=self.embed_len,
            embed_dim=self.embed_dim,
            freeze_IMAGE_queries=self.freeze_IMAGE_queries,
        )

    def save_model(self, output_dir: str):
        logger.info(f"saving `IMAGEEmbedding`...")
        torch.save(self.state_dict(), os.path.join(output_dir, f"{self.save_model_name}.bin"))

    def load_model(self, output_dir: str):
        if check_path_and_file(output_dir, f"{self.save_model_name}.bin"):
            logger.info(f">>> loading `IMAGEEmbedding`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(os.path.join(output_dir, f"{self.save_model_name}.bin"), map_location="cpu"))
            logger.info(f"{loading_status}")
        # HACK: For compatibility
        if check_path_and_file(output_dir, f"IMAGE_queries.pt"):
            self.IMAGE_queries = torch.load(os.path.join(output_dir, "IMAGE_queries.pt"), map_location="cpu")

    def forward(self, batch_size: int = 1):
        return self.IMAGE_queries.repeat(batch_size, 1, 1)


class CLIPVisionEmbedding(MultimodalEmbedding):
    def __init__(
        self,
        clip_vision_model_name_or_path: str,
        projector_type: str = "linear",
        projector_depth: int = 1,
        projector_name_or_path: str = None,
        pretrained_model_name_or_path: str | None = None,
        use_additional_post_layernorm: bool = False,
        select_layer: int = -2,
        embed_hidden_size: int = 4096,
        freeze_clip_vision_model: bool = True,
        freeze_embedding_layers: bool = True,
        freeze_projector: bool = False,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.save_model_name = "clip_vision_embedding"
        self.clip_vision_model_name_or_path = clip_vision_model_name_or_path
        self.clip_vision_model_name_or_path = "./models/clip-vit-large-patch14"
        self.projector_type = projector_type
        self.projector_depth = projector_depth
        self.projector_name_or_path = projector_name_or_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.use_additional_post_layernorm = use_additional_post_layernorm
        self.select_layer = select_layer
        self.embed_hidden_size = embed_hidden_size
        self.freeze_clip_vision_model = freeze_clip_vision_model
        self.freeze_embedding_layers = freeze_embedding_layers
        self.freeze_projector = freeze_projector

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            self.clip_vision_model_name_or_path, local_files_only=local_files_only
        )
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            self.clip_vision_model_name_or_path, local_files_only=local_files_only
        )

        projector_cfg = dict(
            projector=projector_type,
            freeze_projector=freeze_projector,
            depth=projector_depth,
            save_model_name=self.save_model_name,
            model_name_or_path=None,
        )
        self.projector = build_projector(
            projector_cfg, in_hidden_size=self.clip_vision_model.config.hidden_size, out_hidden_size=embed_hidden_size, bias=True
        )

        self._init_weights(self.projector)

        self.post_layernorm = (
            nn.LayerNorm(embed_hidden_size, eps=self.clip_vision_model.config.layer_norm_eps)
            if use_additional_post_layernorm
            else nn.Identity()
        )

        self.image_embed_len = (self.clip_vision_model.config.image_size // self.clip_vision_model.config.patch_size) ** 2

        if pretrained_model_name_or_path is not None:
            self.load_model(pretrained_model_name_or_path)

        if self.projector.load_model(projector_name_or_path):
            logger.info(f">>> loading `CLIPVisionEmbedding` projector from {projector_name_or_path}")

        self.clip_vision_model.requires_grad_(not freeze_clip_vision_model)
        if not freeze_clip_vision_model:
            for i in range(select_layer + 1, 0):
                self.clip_vision_model.vision_model.encoder.layers[i].requires_grad_(False)
            self.clip_vision_model.vision_model.post_layernorm.requires_grad_(False)

        # NOTE: must be set after `self.clip_vision_model.requires_grad_(not freeze_clip_vision_model)`
        self.clip_vision_model.vision_model.embeddings.requires_grad_(not freeze_embedding_layers)

        self.projector.requires_grad_(not freeze_projector)

    @property
    def processor(self):
        return self.clip_image_processor

    @property
    def embed_len(self):
        return self.image_embed_len

    @property
    def embed_dim(self):
        return self.embed_hidden_size

    @property
    def config(self) -> dict:
        return dict(
            clip_vision_model_name_or_path=self.clip_vision_model_name_or_path,
            clip_vision_model_config=self.clip_vision_model.config.to_dict(),
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            select_layer=self.select_layer,
            embed_len=self.embed_len,
            embed_dim=self.embed_dim,
            freeze_clip_vision_model=self.freeze_clip_vision_model,
            freeze_embedding_layers=self.freeze_embedding_layers,
            freeze_projector=self.freeze_projector,
        )

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        if self.freeze_clip_vision_model:
            ignored_modules.append(self.clip_vision_model)
        if self.freeze_projector:
            ignored_modules.append(self.projector)
        return ignored_modules

    def save_model(self, output_dir: str):
        logger.info(f"saving `CLIPVisionEmbedding`...")
        torch.save(self.state_dict(), os.path.join(output_dir, f"{self.save_model_name}.bin"))

    def load_model(self, output_dir: str):
        if check_path_and_file(output_dir, f"{self.save_model_name}.bin"):
            logger.info(f">>> loading `CLIPVisionEmbedding`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(os.path.join(output_dir, f"{self.save_model_name}.bin"), map_location="cpu"))
            logger.info(f"{loading_status}")
        # HACK: For compatibility
        elif check_path_and_file(output_dir, f"clip_vision_model_projector.pt"):
            self.projector.load_state_dict(
                torch.load(os.path.join(output_dir, "clip_vision_model_projector.pt"), map_location="cpu")
            )
        else:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=output_dir, filename=f"{self.save_model_name}.bin")
            logger.info(f">>> loading `CLIPVisionEmbedding`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info(f"{loading_status}")

    def forward(self, images: torch.FloatTensor | None = None):
        # HACK: dummy forward to avoid `find_unused_parameters` error
        is_dummy = images is None
        if is_dummy:
            c, h, w = 3, self.processor.crop_size["height"], self.processor.crop_size["width"]
            images = torch.zeros(1, c, h, w, device=self.clip_vision_model.device, dtype=self.clip_vision_model.dtype)

        output = self.clip_vision_model(images, output_hidden_states=True)
        hidden_state = output.hidden_states[self.select_layer]
        image_features = hidden_state[:, 1:]

        image_embeds = self.projector(image_features)[-1]
        image_embeds = self.post_layernorm(image_embeds)

        if is_dummy:
            return (0.0 * image_embeds).sum()
        else:
            return image_embeds


# head modules
class StableDiffusionHead(MultimodalHead):
    def __init__(
        self,
        diffusion_name_or_path: str,
        projector_type="linear",
        projector_depth: int = 1,
        projector_name_or_path: str = None,
        pretrained_model_name_or_path: str = None,
        embed_hidden_size: int = 4096,
        drop_prob: float | None = None,
        noise_offset: float = 0.0,
        input_perturbation: float = 0.0,
        snr_gamma: float | None = None,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        freeze_vae: bool = True,
        freeze_unet: bool = True,
        freeze_projector: bool = False,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.save_model_name = "stable_diffusion_head"
        diffusion_name_or_path = "./models/stable-diffusion-2-1-base"
        self.diffusion_name_or_path = diffusion_name_or_path
        self.projector_type = projector_type
        self.projector_depth = projector_depth
        self.projector_name_or_path = projector_name_or_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.embed_hidden_size = embed_hidden_size
        self.drop_prob = drop_prob  # Recommended value is 0.1
        self.noise_offset = noise_offset
        self.input_perturbation = input_perturbation  # Recommended value is 0.1
        self.snr_gamma = snr_gamma  # Recommended value is 5.0
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.freeze_vae = freeze_vae
        self.freeze_unet = freeze_unet
        self.freeze_projector = freeze_projector

        self.vae = AutoencoderKL.from_pretrained(diffusion_name_or_path, subfolder="vae", local_files_only=local_files_only)
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            diffusion_name_or_path, subfolder="unet", local_files_only=local_files_only
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            diffusion_name_or_path, subfolder="scheduler", local_files_only=local_files_only
        )
        projector_cfg = dict(
            projector=projector_type,
            freeze_projector=freeze_projector,
            depth=projector_depth,
            save_model_name=self.save_model_name,
            model_name_or_path=None,
        )
        self.projector = build_projector(
            projector_cfg, in_hidden_size=embed_hidden_size, out_hidden_size=self.unet.config.cross_attention_dim, bias=False
        )
        self._init_weights(self.projector)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        if pretrained_model_name_or_path is not None:
            self.load_model(pretrained_model_name_or_path)
            
        if self.projector.load_model(projector_name_or_path):
            logger.info(">>> loading `StableDiffusionHead` projector from {projector_name_or_path}")

        self.vae.requires_grad_(not freeze_vae)
        self.unet.requires_grad_(not freeze_unet)
        self.projector.requires_grad_(not freeze_projector)

    @property
    def processor(self):
        return transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
                transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @property
    def config(self):
        return dict(
            diffusion_name_or_path=self.diffusion_name_or_path,
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            embed_hidden_size=self.embed_hidden_size,
            drop_prob=self.drop_prob,
            noise_offset=self.noise_offset,
            input_perturbation=self.input_perturbation,
            snr_gamma=self.snr_gamma,
            freeze_vae=self.freeze_vae,
            freeze_unet=self.freeze_unet,
            freeze_projector=self.freeze_projector,
        )

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        if self.freeze_vae:
            ignored_modules.append(self.vae)
        if self.freeze_unet:
            ignored_modules.append(self.unet)
        if self.freeze_projector:
            ignored_modules.append(self.projector)
        return ignored_modules

    def save_model(self, output_dir: str):
        logger.info(f"saving `StableDiffusionHead`...")
        torch.save(self.state_dict(), os.path.join(output_dir, f"{self.save_model_name}.bin"))

    def load_model(self, output_dir: str):
        if check_path_and_file(output_dir, f"{self.save_model_name}.bin"):
            logger.info(f">>> loading `StableDiffusionHead`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(os.path.join(output_dir, f"{self.save_model_name}.bin"), map_location="cpu"))
            logger.info(f"{loading_status}")
            # ckpt = torch.load(os.path.join(output_dir, f"{self.save_model_name}.bin"), map_location="cpu")
            # self.projector.projector.weight.data = ckpt["projector.projector.weight"].float()
        # HACK: For compatibility
        elif check_path_and_file(output_dir, f"unet_projector.pt"):
            self.projector.load_state_dict(torch.load(os.path.join(output_dir, "unet_projector.pt"), map_location="cpu"))
        else:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=output_dir, filename=f"{self.save_model_name}.bin")
            logger.info(f">>> loading `StableDiffusionHead`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info(f"{loading_status}")

    def _compute_snr(self, timesteps):
        """
        Computes SNR as per
        https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def forward(
        self,
        images: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        u_encoder_hidden_states: torch.FloatTensor | None = None,
        IMAGE_embeddings: torch.FloatTensor | None = None,
    ):
        # HACK: avoid `find_unused_parameters` error
        is_dummy = images == None
        if is_dummy:
            assert IMAGE_embeddings is not None, "You must provide `IMAGE_embeddings` when dummy forward."
            dummy_image_features = torch.zeros(
                1, IMAGE_embeddings.shape[1], self.embed_hidden_size, device=self.device, dtype=self.dtype
            )
            dummy_image_features = self.projector(dummy_image_features)[-1]
            return (0.0 * dummy_image_features).sum() + (0.0 * IMAGE_embeddings).sum()

        # Convert images to latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        assert (
            encoder_hidden_states.shape[0] == latents.shape[0]
        ), f"encoder_hidden_states.shape[0]: {encoder_hidden_states.shape[0]} != latents.shape[0]: {latents.shape[0]}"
        bsz = latents.shape[0]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn((bsz, latents.shape[1], 1, 1), device=latents.device)
        if self.input_perturbation:
            new_noise = noise + self.input_perturbation * torch.randn_like(noise)

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Train with classifier free guidance, see https://arxiv.org/abs/2207.12598
        if u_encoder_hidden_states is not None and self.drop_prob is not None:
            # u_encoder_hidden_states = self.projector(u_encoder_hidden_states)
            mask = torch.bernoulli(torch.zeros(bsz) + self.drop_prob).to(latents.device)
            mask = mask[:, None, None]
            encoder_hidden_states = (1.0 - mask) * encoder_hidden_states + mask * u_encoder_hidden_states

        # Get the text embedding for conditioning
        encoder_hidden_states = self.projector(encoder_hidden_states)[-1]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self._compute_snr(timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        if is_dummy:
            loss = 0.0 * loss

        return loss

    def check_inputs(
        self,
        height,
        width,
        callback_steps,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}."
            )

        if prompt_embeds is None:
            raise ValueError("Provide `prompt_embeds`. Cannot leave `prompt_embeds` undefined.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}.")

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def _rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    @torch.no_grad()
    def pipeline(
        self,
        # prompt: str | list[str] | MultimodalContent = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        # negative_prompt: str | list[str] | MultimodalContent | None = None,
        num_images_per_prompt: int | None = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: Literal["latent", "pt", "np", "pil"] | None = "pil",
        callback: Callable[[int, int, torch.FloatTensor], None] | None = None,
        callback_steps: int = 1,
        cross_attention_kwargs: dict[str, Any] | None = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # NOTE: The decoder only considers embedding inputs, so there is no raw prompt.
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(height, width, callback_steps, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        batch_size = prompt_embeds.shape[0]

        device = self.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # NOTE: After mounting to LLM, LLM takes on the task.
        assert prompt_embeds is not None, "`prompt_embeds` must be provided by LLM."
        prompt_embeds = self.projector(prompt_embeds)[-1]

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            assert (
                negative_prompt_embeds is not None
            ), "When using classifier free guidance, `negative_prompt_embeds` must be provided by LLM."
            negative_prompt_embeds = self.projector(negative_prompt_embeds)[-1]
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = self._rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        return image
