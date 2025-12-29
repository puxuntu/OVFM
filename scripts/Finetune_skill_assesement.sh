EXP_NAME="finetune_skill_assesement"
DATASET="skillassesementvideos"
CHECKPOINT="../OVFM/checkpoints/pretrain_foundation_base/checkpoint.pth"


if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi


python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  Downstream_tasks/finetune_skill_assesement.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 20 \
  --lr 0.0005 \
  --batch_size_per_gpu 1 \
  --num_workers 4 \
  --num_labels 2 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/$EXP_NAME" \
  --datasetpath "../OVFM/data/downstream/surgical_skill_assessment/" \
  --freeze_backbone False \
#   --opts \