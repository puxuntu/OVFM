EXP_NAME="finetune_tool_existence "
DATASET="toolExistenceImages"
CHECKPOINT="../OVFM/checkpoints/pretrain_foundation_base/checkpoint.pth"

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi


python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="$RANDOM" \
  Downstream_tasks/finetune_tool_existence.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 20 \
  --lr 0.0005 \
  --batch_size_per_gpu 8 \
  --num_workers 4 \
  --num_tools 21 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/$EXP_NAME" \
  --datasetpath "../OVFM/data/downstream/surgical_tool_existence_recognition/" \
  --freeze_backbone False