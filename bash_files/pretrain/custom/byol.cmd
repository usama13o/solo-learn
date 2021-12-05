python C:\Users\Usama\solo-learn\main_pretrain.py  --dataset custom  --backbone swin_large --data_dir F:\\Data\\test\\train  --train_dir cls2  --max_epochs 400  --gpus 0  --accelerator gpu --precision 16  --optimizer sgd  --lars  --grad_clip_lars  --eta_lars 0.02  --exclude_bias_n_norm  --scheduler warmup_cosine  --lr 1.0  --classifier_lr 0.1  --weight_decay 1e-5  --batch_size 64 --num_workers 4  --brightness 0.4  --contrast 0.4  --saturation 0.2  --hue 0.1  --color_jitter_prob 0.8  --gray_scale_prob 0.2  --horizontal_flip_prob 0.5  --gaussian_prob 1.0 0.1  --solarization_prob 0.0 0.2  --num_crops_per_aug 1 1  --name CustomNNClr-400ep-custom  --entity unitn-mhug  --project solo-learn  --save_checkpoint  --method nnclr  --temperature 0.2  --proj_hidden_dim 2048  --pred_hidden_dim 4096  --proj_output_dim 256  --queue_size 1024

