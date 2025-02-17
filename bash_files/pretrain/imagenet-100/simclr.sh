python /home/uz1/projects/solo-learn/main_pretrain.py \
    --dataset wss\
    --backbone swin_large\
    --data_dir /home/uz1/data/wsss/train\
    --train_dir 1.training \
    --max_epochs 400 \
    --gpus 0,1 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 16 \
    --num_workers 4 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --num_crops_per_aug 2 \
    --name simclr-400ep-wss\
    --entity unitn-mhug \
    --project solo-learn \
    --save_checkpoint \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048

# python3 ../../../main_pretrain.py \
#     --dataset imagenet100 \
#     --backbone resnet18 \
#     --data_dir /datasets \
#     --train_dir imagenet-100/train \
#     --val_dir imagenet-100/val \
#     --max_epochs 400 \
#     --gpus 0,1 \
#     --accelerator gpu \
#     --strategy ddp \
#     --sync_batchnorm \
#     --precision 16 \
#     --optimizer sgd \
#     --lars \
#     --grad_clip_lars \
#     --eta_lars 0.02 \
#     --exclude_bias_n_norm \
#     --scheduler warmup_cosine \
#     --lr 0.3 \
#     --weight_decay 1e-4 \
#     --batch_size 128 \
#     --num_workers 4 \
#     --brightness 0.8 \
#     --contrast 0.8 \
#     --saturation 0.8 \
#     --hue 0.2 \
#     --crop_size 224 96 \
#     --num_crops_per_aug 2 6 \
#     --name multicrop-simclr-400ep-imagenet100 \
#     --dali \
#     --entity unitn-mhug \
#     --project solo-learn \
#     --wandb \
#     --save_checkpoint \
#     --method simclr \
#     --proj_hidden_dim 2048 \
#     --temperature 0.1

# python3 ../../../main_pretrain.py \
#     --dataset imagenet100 \
#     --backbone resnet18 \
#     --data_dir /datasets \
#     --train_dir imagenet-100/train \
#     --val_dir imagenet-100/val \
#     --max_epochs 400 \
#     --gpus 0,1 \
#     --accelerator gpu \
#     --strategy ddp \
#     --sync_batchnorm \
#     --precision 16 \
#     --optimizer sgd \
#     --optimizer sgd \
#     --lars \
#     --grad_clip_lars \
#     --eta_lars 0.02 \
#     --exclude_bias_n_norm \
#     --scheduler warmup_cosine \
#     --lr 0.3 \
#     --weight_decay 1e-4 \
#     --batch_size 128 \
#     --num_workers 4 \
#     --brightness 0.8 \
#     --contrast 0.8 \
#     --saturation 0.8 \
#     --hue 0.2 \
#     --crop_size 224 96 \
#     --num_crops_per_aug 2 6 \
#     --name multicrop-supervised-simclr-400ep-imagenet100 \
#     --dali \
#     --entity unitn-mhug \
#     --project solo-learn \
#     --wandb \
#     --save_checkpoint \
#     --method simclr \
#     --temperature 0.1 \
#     --proj_hidden_dim 2048 \
#     --supervised
