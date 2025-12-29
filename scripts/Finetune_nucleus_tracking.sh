EXP_NAME="finetune_nucleus_tracking"
DATASET="nucleus_dataset"
CHECKPOINT="../OVFM/checkpoints/pretrain_foundation_base/checkpoint.pth"


if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi


python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  Downstream_tasks/finetune_nucleus_tracking.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 30 \
  --lr 0.005 \
  --batch_size_per_gpu 8 \
  --num_workers 4 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/$EXP_NAME" \
  --datasetpath "../OVFM/data/downstream/complications_detection/" \
  --freeze_backbone False \
  --iou_weight 0.3 \
  --coord_weight 10.0 \
  --dim_weight 10.0
#   --opts \