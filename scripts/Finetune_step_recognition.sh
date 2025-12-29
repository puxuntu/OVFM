EXP_NAME="finetune_step_recognition"
DATASET="phasedatasetimages"
CHECKPOINT="../OVFM/checkpoints/pretrain_foundation_base/checkpoint.pth"


if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="$RANDOM" \
  Downstream_tasks/finetune_step_recognition.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 20 \
  --lr 0.001 \
  --batch_size_per_gpu 8 \
  --num_workers 4 \
  --num_labels 4 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/$EXP_NAME" \
  --datasetpath "data/downstream/step_recognition/" \
  --freeze_backbone False \
  --opts \
