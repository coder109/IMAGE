from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, HfArgumentParser, TrainingArguments
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image
import argparse
import json
from collections import defaultdict
import numpy as np

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from torchvision.transforms import functional as F

import os

from sentence_transformers import SentenceTransformer

from dataclasses import dataclass, field

import random

seed = 316 # Amen!
random.seed(seed)
BOX_SCALE = 1024


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    hf_hub_model_id: str = field(
        default=None, metadata={"help": "HuggingFace repo to save model weights to"}
    )
    hf_hub_aesthetic_model_id: str = field(
        default=None,
        metadata={"help": "HuggingFace model ID for aesthetic scorer model weights"},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default=None,
        metadata={"help": "HuggingFace model filename for aesthetic scorer model weights"},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})

def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def create_file_if_not_exist(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'a').close()    

def load_model(args, script_args=None):
    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.IMAGE_model)
    return pipeline, tokenizer

# Reference: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/data/datasets/list_dataset.py
class RewardImageDataset(Dataset):
    def __init__(self, images, transforms=None):
        # We need to ensure that: Images are passed in as tensors.
        self.images = images
        self.transforms = transforms
        self.to_tensor_transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        pil_img = T.ToPILImage()(image).convert("RGB")
        w, h = pil_img.size

        target = BoxList([[0, 0, w, h]], pil_img.size, mode="xyxy")
        target = target.clip_to_image(remove_empty=True)
        #boxes = torch.as_tensor(target).reshape(-1, 4)

        image = self.to_tensor_transform(pil_img)

        if self.transforms:
            image = self.transforms(image)
        return image, target, idx

    def get_img_info(self, idx):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        pil_img = T.ToPILImage()(self.images[idx])
        w, h = pil_img.size
        return {"height": h,"width": w}

class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target



# Reference: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/d9f93e847a66e2be7e8e5b965e90adc26e0bf002/maskrcnn_benchmark/data/datasets/visual_genome.py
class VGDataset_NoFile(Dataset):
    def __init__(self, images, transforms=None, custom_eval=True):
        #self.get_custom_imgs(custom_path)
        self.images = images # Pass in as tensors
        index = 0
        self.custom_eval = custom_eval
        self.transforms = transforms
        self.to_tensor_transform = T.Compose([T.ToTensor()])
        self.to_pil_transform = T.ToPILImage()

        # Get Img Info
        self.img_info = []
        for img in self.images:
            pil_img = self.to_pil_transform(img)
            w, h = pil_img.size
            self.img_info.append({'width':int(w), 'height':int(h)})
            index += 1

    def __getitem__(self, index):
        img = self.images[index]
        # Change img to Float Tensor
        #img = img.to(torch.float)
        img = self.to_pil_transform(img).convert("RGB")
        target = torch.Tensor([-1.0])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return F.to_tensor(img), target, index

    def __len__(self):
        return len(self.images)

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:,2]
            new_xmax = w - box[:,0]
            box[:,0] = new_xmin
            box[:,2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy') # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy() # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)
        
        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)

    def get_img_info(self, index):
        return self.img_info[index]

def reward_fn(VSG_model, sim_model, topk=30, DETECTED_SGG_DIR="./", cfg=None):
    # LSG import
    import LSG.sng_parser as lsp
    # VSG import
    from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
    import torch
    from torch import nn
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.data import make_data_loader
    from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler
    from maskrcnn_benchmark.engine.inference import inference
    from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
    from maskrcnn_benchmark.utils.collect_env import collect_env_info
    from maskrcnn_benchmark.utils.comm import synchronize, get_rank
    from maskrcnn_benchmark.utils.logger import setup_logger
    from maskrcnn_benchmark.utils.miscellaneous import mkdir
    from maskrcnn_benchmark.engine.inference import compute_on_dataset, custom_sgg_post_precessing, _accumulate_predictions_from_multiple_gpus
    from maskrcnn_benchmark.data.collate_batch import BatchCollator, BBoxAugCollator
    from tqdm import tqdm
    from sentence_transformers import util

    # Check if we can enable mixed-precision via apex.amp
    try:
        from apex import amp
    except ImportError:
        raise ImportError('Use APEX for mixed precision via apex.amp')
    # Basic import
    import json
    import os

    # LSG
    def parse_single_sentence(sentence):
        graph = lsp.parse(sentence)
        entity_name_blocks = graph["entities"]
        en = []
        for enb in entity_name_blocks:
            en.append(enb["span"])
        rel_blocks = graph["relations"]
        res_block = []
        for rb in rel_blocks:
            res_block.append({"h": en[int(rb["subject"])],
            "r": rb["relation"],
            "t": en[int(rb["object"])]})
        return res_block

    # VSG
    def remove_same_dict_from_list(l_o_d):
        unique_data = [dict(t) for t in {tuple(d.items()) for d in l_o_d}]
        return unique_data

    def return_score(d):
        return d["score"]

    def return_idx(l):
        return int(l[0]["file_idx"])

    def load_info(dict_file, add_bg=True):
        """
        Loads the file containing the visual genome label meanings
        """
        info = json.load(open(dict_file, 'r'))
        if add_bg:
            info['label_to_idx']['__background__'] = 0
            info['predicate_to_idx']['__background__'] = 0
            info['attribute_to_idx']['__background__'] = 0

        class_to_ind = info['label_to_idx']
        predicate_to_ind = info['predicate_to_idx']
        attribute_to_ind = info['attribute_to_idx']
        ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
        ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
        ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

        custom_data_info = {}
        custom_data_info['ind_to_classes'] = ind_to_classes
        custom_data_info['ind_to_predicates'] = ind_to_predicates

        return custom_data_info

    def extract_VSG_triplets(detect_result, topk):
        result = []

        # Load VSG File
        custom_prediction = detect_result
        custom_data_info = load_info(os.path.join(DETECTED_SGG_DIR, "VG-SGG-dicts-with-attri.json"))
        ind_to_classes = custom_data_info['ind_to_classes']
        ind_to_predicates = custom_data_info['ind_to_predicates']
        #idx_to_files = custom_data_info["idx_to_files"]

        # Iterate over all images
        #img_num = len(custom_data_info["idx_to_files"])
        img_num = len(custom_prediction)
        for image_idx in tqdm(range(img_num)):
            res_block = []
            boxes = custom_prediction[int(image_idx)]['bbox'][:]
            box_labels = custom_prediction[int(image_idx)]['bbox_labels'][:]
            box_scores = custom_prediction[int(image_idx)]['bbox_scores'][:]
            all_rel_labels = custom_prediction[int(image_idx)]['rel_labels']
            all_rel_scores = custom_prediction[int(image_idx)]['rel_scores']
            all_rel_pairs = custom_prediction[int(image_idx)]['rel_pairs']

            for i in range(len(box_labels)):
                box_labels[i] = ind_to_classes[box_labels[i]]

            for i in range(len(all_rel_pairs)):
                if ind_to_predicates[all_rel_labels[i]] == "has":
                    continue
                res_block.append({"h": box_labels[all_rel_pairs[i][0]],
                "r": ind_to_predicates[all_rel_labels[i]],
                "t": box_labels[all_rel_pairs[i][1]],
                "score": all_rel_scores[i],
                #"file_idx": os.path.split(idx_to_files[image_idx])[-1].split(".")[0]
                })
                res_block.sort(key=return_score, reverse=True)
                res_block = res_block[:topk]
            result.append(remove_same_dict_from_list(res_block))
            #result.sort(key=return_idx)
        return result

    # Sim Calc
    def encode_elem(model, elem):
        return model.encode(elem, convert_to_tensor=True)

    def encode_triplet(model, triplet):
        return torch.stack((encode_elem(model, triplet["h"]), encode_elem(model, triplet["r"]), encode_elem(model, triplet["t"])))

    def sim_calc(model, src_triplet, tgt_triplet):
        src_emb, tgt_emb = encode_triplet(model, src_triplet), encode_triplet(model, tgt_triplet)
        cos = nn.CosineSimilarity(dim=1)
        sim = cos(src_emb, tgt_emb)
        sim = torch.mean(sim)
        return sim

    def get_one_sim(model, LSG_list, VSG_list):
        LSG, VSG = LSG_list, VSG_list
        lsg_num = len(LSG)
        block_best_score = 0.00
        # Iterate over LSG blocks
        for lsg in LSG:
            curr_best_score = 0.00
            # Iterate over VSG blocks and update curr_best_score
            for vsg in VSG:
                sim = sim_calc(model, lsg, vsg)
                curr_best_score = curr_best_score if curr_best_score >= sim.item() else sim.item()
            block_best_score += curr_best_score
        return 0.0 if block_best_score == 0.0 else block_best_score / lsg_num

    def _fn(images, prompts, metadata):
        # Parse LSG
        LSG_list = [parse_single_sentence(p) for p in prompts]
        # Parse VSG
        # Load data
        img_dataset = VGDataset_NoFile(images)
        sampler = make_data_sampler(img_dataset, False, False)
        aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        images_per_gpu = images_per_batch
        num_iters = None
        start_iter = 0
        batch_sampler = make_batch_data_sampler(img_dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter)
        collator = BBoxAugCollator() 
        img_data = DataLoader(img_dataset, batch_size=1, shuffle=False, collate_fn=collator, batch_sampler=batch_sampler)
        #img_data = make_data_loader(cfg=cfg, mode="test", dataset_to_test=cfg.DATASETS.TO_TEST)
        # Predict
        predictions = compute_on_dataset(VSG_model, img_data, device, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER, timer=None)
        predictions = _accumulate_predictions_from_multiple_gpus(predictions, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)
        # Get Triplets
        detected_sgg = custom_sgg_post_precessing(predictions)
        # For Debug Use
        json.dump(detected_sgg, open("log.json", "a"))
        VSG_list = extract_VSG_triplets(detected_sgg, topk)
        # Calculate Similarity Value as final result
        tmp_val = get_one_sim(sim_model, LSG_list[0], VSG_list[0])
        print(LSG_list)
        print(VSG_list)
        '''
        if tmp_val == 0.:
            curr_best = 0.
            for VSG in VSG_list[0]:
                curr_best = curr_best if curr_best >= VSG['score'] else VSG['score']
            return torch.tensor([curr_best]), {}
        '''
        return torch.tensor([get_one_sim(sim_model, LSG_list[0], VSG_list[0])]), {}

    return _fn

def prompt_fn(file_path):
    def _fn():
        blocks = json.load(open(file_path))
        length = len(blocks)
        return blocks[random.randint(0, length-1)]["src"], {}
    return _fn


def main(args, detect_model, cfg):
    script_cfg = ScriptArguments(
        pretrained_model=args.sd_model,
    )

    train_cfg = DDPOConfig(
        num_epochs = args.num_epochs,
        logdir = args.logging_dir,
        train_batch_size = args.train_batch_size,
        sample_batch_size = args.sample_batch_size,
        train_learning_rate = args.train_learning_rate,
    )
    train_cfg.project_kwargs = {
        "logging_dir": args.logging_dir,
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": args.ckpt_dir
    }

    pipe, tokenizer = load_model(args, script_cfg)

    sim_model = SentenceTransformer(args.sim_model)

    trainer = DDPOTrainer(
        train_cfg,
        prompt_function=prompt_fn(args.dataset),
        reward_function=reward_fn(detect_model, sim_model, args.topk, cfg.DETECTED_SGG_DIR, cfg),  # Use the reward function defined above
        sd_pipeline=pipe,
    )

    trainer.train()

    #trainer.save_model(args.ckpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model-related
    parser.add_argument(
        "--sd_model",
        type=str,
        default=None,
        help="Path to the Stable Diffusion model",
    )
    parser.add_argument(
        "--IMAGE_model",
        type=str,
        default=None,
        help="Path to the DreamLLM model",
    )
    parser.add_argument(
        "--sim_model",
        type=str,
        default=None,
        help="Path to the similarity model",
    )
    # VSG-related
    parser.add_argument(
        "--detect_model",
        type=str,
        default=None,
        help="Path to the object detection model",
    )
    parser.add_argument(
        "--detect_cfg",
        type=str,
        default=None,
        help="Path to the object detection config file",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Number of triplets to return",
    )
    # Dataset-related
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the dataset. It should be in JSON format. Each line should contain prompt, img, score.",
    )
    # Image-related
    parser.add_argument(
        "--img_storage_path",
        type=str,
        default=None,
        help="Path to the image storage directory",
    )
    # Training Config
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="Path to the logging directory",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=16,
        help="Sampling batch size",
    )
    parser.add_argument(
        "--train_learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    # VSG Config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    create_dir_if_not_exist(args.img_storage_path)
    cfg.merge_from_file(args.detect_cfg)
    cfg.merge_from_list(args.opts)
    detect_model = build_detection_model(cfg)
    detect_model.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, detect_model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    
    main(args, detect_model, cfg)
