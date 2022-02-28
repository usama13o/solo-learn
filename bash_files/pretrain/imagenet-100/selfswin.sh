# python /home/uz1/projects/solo-learn/main_pretrain.py \
#     --dataset custom\
#     --backbone swin_large\
#     --data_dir /data1/uz1 \
#     --train_dir peso \
#     --max_epochs 200 \
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
#     --lr 0.4 \
#     --weight_decay 1e-5 \
#     --batch_size 16\
#     --brightness 0.4 \
#     --contrast 0.4 \
#     --saturation 0.2 \
#     --hue 0.1 \
#     --gaussian_prob 1.0 0.1 \
#     --solarization_prob 0.0 0.2 \
#     --num_crops_per_aug 1 1 \
#     --num_workers 16 \
#     --name selfswin-200ep-peso-modDepth\
#     --entity unitn-mhug \
#     --project solo-learn \
#     --save_checkpoint \
#     --method selfswin \
#     --temperature 0.2 \
#     --proj_hidden_dim 2048 \
#     --pred_hidden_dim 4096 \
#     --proj_output_dim 256 \
#     --queue_size 4096

python /home/uz1/projects/solo-learn/main_pretrain.py \
    --dataset svs_h5\
    --backbone swin_large\
    --data_dir /home/uz1/data/tupa \
    --train_dir patches \
    --max_epochs 200 \
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
    --num_workers 16 \
    --name selfswin-200ep-peso-modDepth\
    --entity unitn-mhug \
    --project solo-learn \
    --save_checkpoint \
    --method selfswin \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --pred_hidden_dim 4096 \
    --proj_output_dim 256 \
    --queue_size 4096
