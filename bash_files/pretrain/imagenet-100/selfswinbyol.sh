python ../../../main_pretrain.py \
    --dataset custom\
    --backbone swin_large\
    --data_dir /data1/uz1/\
    --train_dir peso \
    --max_epochs 60 \
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
    --lr 0.4 \
    --weight_decay 1e-5 \
    --batch_size 16\
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --num_workers 4 \
    --name selfswin-60ep-Custom \
    --entity unitn-mhug \
    --project solo-learn \
    --save_checkpoint \
    --method selfswinbyol \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --pred_hidden_dim 4096 \
    --proj_output_dim 256 \
    --queue_size 1024
