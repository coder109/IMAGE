from trl_sd import VGDataset_NoFile, reward_fn, ScriptArguments
from trl import DDPOConfig,  DefaultDDPOStableDiffusionPipeline
from diffusers import StableDiffusionPipeline
#from trl import DDPOTrainer

from run_clm_llms import (
    ModelArguments,
    DataTrainingArguments,
    SDArguments,
    setup_logging,
    setup_datasets,
    setup_config,
    setup_models,
    preprocess_data,
    #init_training,
    train_fn,
)

import logging

import evaluate, transformers, torch
from tqdm import tqdm
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    is_torch_tpu_available,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from sentence_transformers import SentenceTransformer

from DDPO_Trainer import DDPOTrainer

from transformers import CLIPImageProcessor, CLIPVisionModel
# Import END

import torchvision
import torchvision.transforms as T

import sacrebleu

to_pil_transform = T.ToPILImage()

# Pre-running check BEGIN: aligned with run_clm_llms.py
check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
# Pre-running check END

def get_img_tensor(image, image_processor):
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image_tensor

def load_sd_related_model(reward_fn, prompt_fn, lora_path=None):
    # SD-related Parameters BEGIN
    sd_model = ""
    sim_model = ""
    detect_model = ""
    detect_cfg = ""
    # SD-related Parameters END

    cfg.merge_from_file(detect_cfg)
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
    cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR = "CausalAnalysisPredictor"
    cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE = "TDE"
    cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE = "sum"
    cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER = "motifs"
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.TEST.CUSTUM_EVAL = True
    cfg.DTYPE = "float16"
    cfg.GLOVE_DIR = ""
    cfg.MODEL.PRETRAINED_DETECTOR_CKPT = ""
    cfg.OUTPUT_DIR = ""
    cfg.DETECTED_SGG_DIR = ""

    detect_model = build_detection_model(cfg)
    detect_model.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, detect_model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    # MOD BEGIN
    script_cfg = ScriptArguments(
        pretrained_model=sd_model,
    )

    '''
    train_cfg = DDPOConfig(
        num_epochs = ddpo_num_epochs,
        logdir = ddpo_logging_dir,
        train_batch_size = ddpo_train_batch_size,
        sample_batch_size = ddpo_sample_batch_size,
        train_learning_rate = ddpo_train_learning_rate,

    )
    train_cfg.project_kwargs = {
        "logging_dir": ddpo_logging_dir,
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": ddpo_ckpt_dir
    }
    '''

    pipe = StableDiffusionPipeline.from_pretrained(
        script_cfg.pretrained_model,
    ).to("cuda")
    if lora_path:
        pipe.unet.load_attn_procs(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="triplet")
    #pipe.set_progress_bar_config(disable=True)

    sim_mod = SentenceTransformer(sim_model)

    '''
    trainer = DDPOTrainer(
        train_cfg,
        prompt_function=prompt_fn(),
        reward_function=reward_fn(detect_model, sim_mod, topk, cfg.DETECTED_SGG_DIR, cfg),  # Use the reward function defined above
        sd_pipeline=pipe,
    )
    '''
    return pipe, sim_mod, detect_model

# Custom Trainer
class CustomTrainer(Trainer):
    def VSG_parameters_injector(self, ddpo_trainer):
        self.ddpo_trainer = ddpo_trainer
    
    def reward_fn_injector(self, reward_fn):
        self.reward_fn = reward_fn

    def sd_pipe_injector(self, pipe):
        self.pipeline = pipe

    def tokenizer_injector(self, tokenizer):
        self.tokenizer = tokenizer

    def generate_image_via_sd(self, prompt):
        '''
        1. We use the SD model from DDPOTrainer to generate the image, not the locally saved one.
           Because we want to use the trained SD to generate the image.
        2. This function will only take one prompt as input, and return one image.
        '''

        '''
        prompt_ids = self.pipeline.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256,
        ).input_ids
        prompt_embed = self.pipeline.text_encoder(prompt_ids)[0]
        sample_neg_prompt_embeds = self.ddpo_trainer.neg_prompt_embed.repeat(
            self.ddpo_trainer.config.sample_batch_size, 
            1, 
            1
        )

        sd_output = self.pipeline(

        )
        '''
        return self.pipeline(            
            height=512,
            width=512,
            prompt=prompt,
            negative_prompt_embeds=None,
            num_inference_steps=50,
            guidance_scale=7.5,
            guidance_rescale=1.0,
            eta=0.0,
            output_type="pt",
            ).images[0]
    
    def compute_rewards_use_trainer(self, prompt_image_pairs, is_async=False):
        if not is_async:
            rewards = []
            for images, prompts, prompt_metadata in prompt_image_pairs:
                images = images.to(self.ddpo_trainer.accelerator.device)
                #prompts = prompts.to(self.ddpo_trainer.accelerator.device)
                reward, reward_metadata = self.ddpo_trainer.reward_fn(images, prompts, prompt_metadata)
                rewards.append(
                    (
                        torch.as_tensor(1. - reward, device=self.ddpo_trainer.accelerator.device),
                        reward_metadata,
                    )
                )
        else:
            rewards = self.ddpo_trainer.executor.map(lambda x: self.ddpo_trainer.reward_fn(*x), prompt_image_pairs)
            rewards = [
                (torch.as_tensor(1. -reward.result(), device=self.ddpo_trainer.accelerator.device), reward_metadata.result())
                for reward, reward_metadata in rewards
            ]

        return zip(*rewards)
    
    def compute_rewards(self, prompt_image_pairs, is_async=False):
        rewards = []
        for images, prompts, prompt_metadata in prompt_image_pairs:
            reward, reward_metadata = self.reward_fn(images, [prompts], prompt_metadata)
            rewards.append((torch.as_tensor(1. - reward), reward_metadata))
        return zip(*rewards)
        

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        '''
        MULTI-GPU NOT SUPPORTED
        '''
        # DEBUG
        #print(inputs)
        # Get input in NLP format.
        input_texts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        prompt_image_pairs = []
        image_list = []
        # Generate corresponding images.
        for input_text in tqdm(input_texts):
            if "?" in input_text:
                in_text = input_text.split("USER:")[-1].split("\nASSISTANT")[0].split("?")[-1].strip()
            else:
                in_text = input_text.split("USER:")[-1].split("\nASSISTANT")[0].split(".")[1].strip()


            image = self.generate_image_via_sd(
                    prompt=in_text,
            )

            pil_image = to_pil_transform(image).convert("RGB")

            image_tensor = get_img_tensor(pil_image, model.model.clip_vision_embedding.processor)
            image_list.append(image_tensor)
            prompt_image_pairs.append((image, in_text, None))
        # Calculate rewards
        rewards = self.compute_rewards(prompt_image_pairs, is_async=False)
        reward_val, _ = rewards
        # Get the LM Loss
        '''
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        else:
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        '''
        # DEBUG
        #print(image_list)
        #print(image_list[0].shape)
        #print(torch.stack(image_list))
        IMAGEllm_output = model.forward(labels=inputs["labels"],
                                      images=torch.stack(image_list).to(inputs["labels"].device),
                                      attention_mask=inputs["attention_mask"],
                                      input_ids=inputs["input_ids"])
        # Get Prediction and decode into NLP
        logits = IMAGEllm_output.logits
        predictions = torch.argmax(logits, dim=-1)
        decoded = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Get only output in var decoded
        decoded = [x.split(":")[-1] for x in decoded]

        # DEBUG
        #print(decoded)

        # Get Loss
        IMAGEllm_loss = IMAGEllm_output.loss

        reward_final = torch.mean(torch.stack(reward_val))
        loss = IMAGEllm_loss / IMAGEllm_loss.detach() + reward_final / reward_final.detach()

        return (loss, None) if return_outputs else loss
             


def init_training(training_args, data_args, lm_datasets, logger, model, tokenizer):
    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        #if "train" not in tokenized_datasets:
        # xxx: 2023-03-14
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # xxx: print samples
        logger.info("xxx: Showcase the tokenized training samples.")
        for i in range(3):
            print(next(iter(train_dataset)))

    if training_args.do_eval:
        #if "validation" not in tokenized_datasets:
        # xxx: 2023-03-14
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        #data_collator=default_data_collator,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                          padding=True, label_pad_token_id=IGNORE_INDEX ),
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )
    return trainer, train_dataset, eval_dataset

def prompt_fn():
    def _fn():
        return
    return _fn

def main():
    # Get args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, SDArguments))
    model_args, data_args, training_args, args = parser.parse_args_into_dataclasses()

    # Set up VSG-related models
    pipe, sim_model, detect_model = load_sd_related_model(reward_fn, prompt_fn)

    # Load CLIP for Image Generation
    #vision_tower = CLIPVisionModel.from_pretrained(args.clip_model_path)
    #image_processor = CLIPImageProcessor.from_pretrained(args.clip_model_path, device_map="auto")

    # Align with run_clm_llms.py
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging BEGIN: aligned with run_clm_llms.py
    last_checkpoint = setup_logging(training_args, logger)

    # Set up datasets
    raw_datasets = setup_datasets(data_args, model_args, logger)

    # Set up config
    config = setup_config(model_args, logger)

    # Set up models
    model, tokenizer = setup_models(model_args, logger, config)

    # Preprocess Data
    lm_datasets = preprocess_data(training_args, data_args, tokenizer, raw_datasets, logger)

    # Init trainer
    llm_trainer, train_dataset, eval_dataset = init_training(training_args, data_args, lm_datasets, logger, model, tokenizer)

    # Inject Tokenizer into our main LM Trainer
    llm_trainer.tokenizer_injector(tokenizer)

    # Inject DDPOTrainer into our main LM Trainer
    #llm_trainer.VSG_parameters_injector(ddpo_trainer)

    # Inject Reward Function into our main LM Trainer
    llm_trainer.reward_fn_injector(reward_fn(detect_model, sim_model, args.topk, cfg.DETECTED_SGG_DIR, cfg=cfg))

    # Inject SD Pipeline into our main LM Trainer
    llm_trainer.sd_pipe_injector(pipe)

    # Train
    train_fn(training_args, model_args, last_checkpoint, data_args, train_dataset, llm_trainer, tokenizer)

if __name__ == "__main__":
    main()