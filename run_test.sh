# 运行 Python 程序
# python "test.py" \
#   --name pix2pix_v100_1 \
#   --model pix2pix \
#   --gpu_ids 0 \
#   --display_port 8115 \
#   --netG unet_128 \
#   --batch_size 1 \
#   --lambda_L1 1 \

# python "test.py" \
#   --name pix2pix_v100_2 \
#   --model pix2pix \
#   --gpu_ids 0 \
#   --display_port 8115 \
#   --netG unet_128 \
#   --batch_size 1 \
#   --lambda_L1 1 \

# polar_perceptual
# python "test.py" \
#   --name polar_perceptual \
#   --model pix2pix \
#   --gpu_ids 1 \
#   --display_port 8001 \
#   --netG unet_128 \
#   --batch_size 1 \
#   --lambda_L1 1 \
#   --input_type polar \

# polar_l1
# python "test.py" \
#   --name polar_l1 \
#   --model pix2pix \
#   --gpu_ids 0 \
#   --display_port 8000 \
#   --netG unet_128 \
#   --batch_size 1 \
#   --lambda_L1 100 \
#   --input_type polar \
#   --loss_type l1 \

# perceptual
# python "test.py" \
#   --name perceptual \
#   --model pix2pix \
#   --gpu_ids 2 \
#   --display_port 8002 \
#   --netG unet_128 \
#   --batch_size 1 \
#   --lambda_L1 0.1 \

# l1
python "test.py" \
  --name l1 \
  --model pix2pix \
  --gpu_ids 0 \
  --display_port 8003 \
  --netG unet_128 \
  --batch_size 1 \
  --lambda_L1 100 \
  --loss_type l1 \