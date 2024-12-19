from modeling.modeling import IMAGELLMForCausalMLM, IMAGEConfig
from stage2_config_generator import config as external_config
from modeling.logger import logger

from transformers import AutoTokenizer


if __name__ == "__main__":
    model_name_or_path = ""
    config = IMAGEConfig().from_pretrained(model_name_or_path)
    plugin_modules_names = []
    for _, plugin_config_init_kwargs in external_config.model.plugins_config_init_kwargs.items():
        name = config.update_plugins(plugin_config_init_kwargs)
        logger.info(f"Successfully update plugin `{name}` with `ConfigAndInitKwargs` to `IMAGELLMConfig`.")
        plugin_modules_names.append(name)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = IMAGELLMForCausalMLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        tokenizer=tokenizer,
    )
    model.get_output_embeddings().requires_grad_(not external_config.model.freeze_lm_head)