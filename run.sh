#!/bin/bash

# 确保脚本在遇到错误时停止执行
set -e

# 运行 Python 程序
python "train.py" \
  --name pix2pix_v100_2 \
  --model pix2pix \
  --gpu_ids 0 \
  --display_port 8115 \
  --netG unet_128 \
  --batch_size 24 \
  --lambda_L1 1 \

# kernprof -l -v "train.py" \
#   --name profile \
#   --model pix2pix \
#   --gpu_ids 1 \
#   --display_port 8115 \
#   --netG unet_128 \
#   --batch_size 1 \
#   --lambda_L1 1 \
#   > output_v3.log