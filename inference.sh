#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py \
--mid_product_image_path "" \
--sd_lora_path "" \
--src_lang_file "" \
--tgt_lang_file "" \
--IMAGE_llm_model_path "" \
--output_file ""
