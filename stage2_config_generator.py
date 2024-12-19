import json
import os

from omegaconf import OmegaConf

from modeling.configuration import ConfigAndInitKwargs, create_config_init_kwargs
from modeling.modeling_plugins import CLIPVisionEmbedding, IMAGEEmbedding, StableDiffusionHead
from modeling.import_utils import is_volc_mlplatform_available

if is_volc_mlplatform_available():
    local_files_only = True
else:
    local_files_only = False

# NOTE: don't forget add `embed_hidden_size` kwargs
IMAGE_embedding_config_init_kwargs = create_config_init_kwargs(
    ConfigAndInitKwargs(
        _class_=IMAGEEmbedding,
        _name_="IMAGE_embedding",
        _plugin_type_="embedding",
        pretrained_model_name_or_path=None,
        num_IMAGE_queries=64,
        # embed_hidden_size=hidden_size,
        freeze_IMAGE_queries=False,
    )
)
clip_vision_embedding_config_init_kwargs = create_config_init_kwargs(
    ConfigAndInitKwargs(
        _class_=CLIPVisionEmbedding,
        _name_="clip_vision_embedding",
        _plugin_type_="embedding",
        projector_type="linear",
        projector_depth=1,
        clip_vision_model_name_or_path="",
        pretrained_model_name_or_path=None,
        # embed_hidden_size=hidden_size,
        use_additional_post_layernorm=False,
        select_layer=-2,
        freeze_clip_vision_model=True,
        freeze_embedding_layers=True,
        freeze_projector=False,
        local_files_only=local_files_only,
    )
)
stable_diffusion_head_config_init_kwargs = create_config_init_kwargs(
    ConfigAndInitKwargs(
        _class_=StableDiffusionHead,
        _name_="stable_diffusion_head",
        _plugin_type_="head",
        projector_type="linear",
        projector_depth=1,
        diffusion_name_or_path="",
        pretrained_model_name_or_path=None,
        # embed_hidden_size=hidden_size,
        freeze_vae=True,
        freeze_unet=True,
        freeze_projector=False,
        local_files_only=local_files_only,
    )
)


config = OmegaConf.create(flags={"allow_objects": True})

model_name_or_path = ""


# load Vision Encoder connector weights
clip_vision_embedding_config_init_kwargs.pretrained_model_name_or_path = model_name_or_path
# load Diffusion Decoder connector weights
stable_diffusion_head_config_init_kwargs.pretrained_model_name_or_path = model_name_or_path
IMAGE_embedding_config_init_kwargs.pretrained_model_name_or_path = model_name_or_path


with open(os.path.join(model_name_or_path, "config.json")) as f:
    model_config = json.load(f)
hidden_size = model_config["hidden_size"]
max_position_embeddings = model_config["max_position_embeddings"]

clip_vision_embedding_config_init_kwargs.pretrained_model_name_or_path = "/"
stable_diffusion_head_config_init_kwargs.pretrained_model_name_or_path = ""
IMAGE_embedding_config_init_kwargs.pretrained_model_name_or_path = ""

IMAGE_embedding_config_init_kwargs.embed_hidden_size = hidden_size
clip_vision_embedding_config_init_kwargs.embed_hidden_size = hidden_size
stable_diffusion_head_config_init_kwargs.embed_hidden_size = hidden_size

# Vision Encoder
clip_vision_embedding_config_init_kwargs.freeze_clip_vision_model = True
clip_vision_embedding_config_init_kwargs.freeze_embedding_layers = True  # freeze all patch, class, and position embeddings
clip_vision_embedding_config_init_kwargs.freeze_projector = True 
# Diffusion Decoder
stable_diffusion_head_config_init_kwargs.freeze_vae = True
stable_diffusion_head_config_init_kwargs.freeze_unet = True
stable_diffusion_head_config_init_kwargs.freeze_projector = False # Not SURE
IMAGE_embedding_config_init_kwargs.freeze_IMAGE_queries = True

config.model = dict(
    model_name_or_path=model_name_or_path,
    model_max_length=max_position_embeddings,
    local_files_only=local_files_only,
    special_tokens_dict={},
    average_init_embed_tokens=False,
    freeze_embed_tokens=True, # NOT SURE
    freeze_lm_model=False,
    freeze_lm_head=True, # NOT SURE
    loss_weight_lm=1.0,
    loss_weight_vm=10.0,
    plugins_config_init_kwargs=dict(
        clip_vision_embedding=clip_vision_embedding_config_init_kwargs,
        IMAGE_embedding=IMAGE_embedding_config_init_kwargs,
        stable_diffusion_head=stable_diffusion_head_config_init_kwargs,
    ),
    use_fast_tokenizer=False,
    use_flash_attention_2=False,
)

config.model.update(model_config)

# print(config)
