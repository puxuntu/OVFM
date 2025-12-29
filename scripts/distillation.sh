############# Backbone: Small ######################

DATA_PATH="./data/pretrain"
EXP_NAME="distillation_train_small"

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="$RANDOM" \
  Distillation/distillation.py \
  --arch "timesformer" \
  --batch_size_per_gpu 8 \
  --data_path "${DATA_PATH}" \
  --output_dir "checkpoints/$EXP_NAME" \
  --student_vit "vit_small" \
  --teacher_model_path "checkpoints/pretrain_foundation_base/checkpoint.pth" \
  --opts \
  MODEL.TWO_STREAM False \
  MODEL.TWO_TOKEN False \
  DATA.NO_FLOW_AUG False \
  DATA.USE_FLOW False \
  DATA.RAND_CONV False \
  DATA.NO_SPATIAL False


############# Backbone: Tiny ######################

# DATA_PATH="./data/pretrain"
# EXP_NAME="distillation_train_tiny"

# if [ ! -d "checkpoints/$EXP_NAME" ]; then
#   mkdir "checkpoints/$EXP_NAME"
# fi

# python -m torch.distributed.run \
#   --nproc_per_node=2 \
#   --master_port="$RANDOM" \
#   Distillation/distillation.py \
#   --arch "timesformer" \
#   --batch_size_per_gpu 8 \
#   --data_path "${DATA_PATH}" \
#   --output_dir "checkpoints/$EXP_NAME" \
#   --student_vit "vit_tiny" \
#   --teacher_model_path "checkpoints/pretrain_foundation_base/checkpoint.pth" \
#   --opts \
#   MODEL.TWO_STREAM False \
#   MODEL.TWO_TOKEN False \
#   DATA.NO_FLOW_AUG False \
#   DATA.USE_FLOW False \
#   DATA.RAND_CONV False \
#   DATA.NO_SPATIAL False