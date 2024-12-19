#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python trl_sd.py \
--sd_model "" \
--IMAGE_model "" \
--sim_model "" \
--detect_model "" \
--detect_cfg "" \
--topk 30 \
--dataset "" \
--img_storage_path "tempo" \
--num_epochs 50 \
--train_batch_size 1 \
--sample_batch_size 1 \
--train_learning_rate 5e-5 \
--logging_dir "./log/" \
--ckpt_dir "./saves/" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
TEST.IMS_PER_BATCH 1 \
TEST.CUSTUM_EVAL True \
DTYPE "float16" \
GLOVE_DIR ./VSG/glove/ \
MODEL.PRETRAINED_DETECTOR_CKPT ./VSG/checkpoint/ \
OUTPUT_DIR ./VSG/checkpoint/ \
DETECTED_SGG_DIR ./tempo
