import omni
import argparse

import base64
import json
import math
import os
from io import BytesIO

import argparse

import pandas as pd
import requests
import torch
from tqdm import tqdm

from PIL import Image

from modeling.configuration import IMAGELLMConfig
from modeling.modeling import IMAGELLMForCausalMLM
from modeling.conversation import KeywordsStoppingCriteria, SeparatorStyle, default_conversation
from modeling.logger import logger

from transformers import CLIPImageProcessor, CLIPVisionModel, LlamaTokenizer

from utils import *


import torchvision.transforms as transforms

device = "cuda"

def generate_input(input_sentence):
       return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
       ### USER:\nPlease translate following sentence from English to German: {input_sentence}"""

def load_model(model_name_or_path, lora_path=None):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path,
        #model_name_or_path,
        padding_side="right",
        device_map="auto",
    )
    tokenizer.pad_token = tokenizer.eos_token
    config = IMAGELLMConfig.from_pretrained(
	model_name_or_path
    )
    config = config.reset_plugins_init_kwargs()
    model = IMAGELLMForCausalMLM.from_pretrained(
        model_name_or_path,
        tokenizer=tokenizer,
        config=config,
        reset_plugin_model_name_or_path=True, # NOTE: Don't forget to reset.
        device_map = "auto",
        torch_dtype=torch.float,
    )
     
    model = torch.compile(model)
    model.stable_diffusion_head.set_progress_bar_config(disable=True)
    if lora_path:
        model.stable_diffusion_head.unet.load_attn_procs(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="triplet")
    return tokenizer, model

def generate(args, tokenizer, conv, image_tensor, model):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    prompt = conv.get_prompt()
    if prompt.endswith(stop_str):
        prompt = prompt[: -len(stop_str)]

    inputs = tokenizer([prompt])
    print(inputs)
    print(tokenizer.batch_decode(inputs['input_ids']))
    input_ids = torch.as_tensor(inputs.input_ids).to(device)

    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=torch.tensor(image_tensor, dtype=torch.float).unsqueeze(0).half().to(device),
            num_beams=2,
            do_sample=True,
            temperature=0.99,
            max_new_tokens=256,
            stopping_criteria=[stopping_criteria],
            top_p=0.9,
            repetition_penalty=1.5,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        logger.warning(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.split("\n")[0].strip()
    print(outputs)
    return outputs

def get_img_tensor(image, image_processor):
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--IMAGE_llm_model_path", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="./models/clip-vit-large-patch14")
    
    parser.add_argument("--external_image_path", type=str, default=None, help="If None, we will generate the images using models instead of using external images.")
    parser.add_argument("--image_split_file", type=str, default=None, help="If None, we will use index from 0 to num_prompts.")
    parser.add_argument("--mid_product_image_path", type=str, default=None, help="If None, we will not save the mid product images.")

    parser.add_argument("--sd_lora_path", type=str, default=None)
    
    parser.add_argument("--src_lang_file", type=str, default=None)
    parser.add_argument("--tgt_lang_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    
    args = parser.parse_args()
    if not os.path.exists(args.mid_product_image_path):
        os.makedirs(args.mid_product_image_path)

    # Load Model
    print("Loading Model...")
    tokenizer, model = load_model(args.IMAGE_llm_model_path, args.sd_lora_path)
    vision_tower = CLIPVisionModel.from_pretrained(args.clip_model_path)
    image_processor = CLIPImageProcessor.from_pretrained(args.clip_model_path, device_map="auto")
    
    # Load Dataset
    print("Loading Datasets...")
    src_contents = load_dataset(args.src_lang_file)
    tgt_contents = load_dataset(args.tgt_lang_file)
    assert len(src_contents) == len(tgt_contents)

    # Inference Main Loop
    answer_list = []
    if args.image_split_file:
        with open(args.image_split_file,"r") as f:
            split_lines = f.readlines()
    print("Start Inference...")
    for curr_idx in tqdm(range(len(src_contents))):
        src_content = src_contents[curr_idx]
        qs = generate_input(src_content)
        
        # Initialize Conversation
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], "")

        # Get Image
        if args.external_image_path:
            if args.image_split_file:
                image = Image.open(os.path.join(args.external_image_path, split_lines[curr_idx].split("#")[0].strip()))
            else:
                try:
                    image = Image.open(os.path.join(args.external_image_path, str(curr_idx) + ".png"))
                except:
                    image = Image.open(os.path.join(args.external_image_path, str(curr_idx) + ".jpg"))
            transform = transforms.Compose([transforms.ToTensor()])

            try:
                image = transform(image)
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            except:
                continue
                image = None

        else:
            image = model.stable_diffusion_pipeline(
                tokenizer=tokenizer,
                prompt=src_content,
            )[0]
            if args.mid_product_image_path:
                image.save(os.path.join(args.mid_product_image_path, str(curr_idx)+".png"))
            image = get_img_tensor(image, image_processor)

        # Get Result
        try:
            output = generate(args, tokenizer, conv, image, model)
        except:
            output = ""



        # Construct Answer Block
        answ_block = construct_result_block(src_content, tgt_contents[curr_idx], str(output))
        answer_list.append(answ_block)
        write_into_jsonl([answ_block], args.output_file)
    # Filter the Output File
    filter_result(args.output_file)
