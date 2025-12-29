DATA_PATH="./data/pretrain"
EXP_NAME="pretrain_foundation_base"

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="$RANDOM" \
  pretrain_train.py \
  --arch "timesformer" \
  --batch_size_per_gpu 4 \
  --data_path "${DATA_PATH}" \
  --output_dir "checkpoints/$EXP_NAME" \
  --vit_type "vit_base" \
  --opts \
  MODEL.TWO_STREAM False \
  MODEL.TWO_TOKEN False \
  DATA.NO_FLOW_AUG False \
  DATA.USE_FLOW False \
  DATA.RAND_CONV False \
  DATA.NO_SPATIAL False