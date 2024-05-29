# python /data/crossmoda2023/query-selected-attention/train.py -n contrast_v9 --model style --netG resnet_9blocks_style --display_id 0 --n_epochs 900 --n_epochs_decay 100 --update_html_freq 100
python -W ignore train.py -n basic_v9 -t Basic -c 4 --max_epoch 2000 --skip_val_epoch 2000 --html_iter 50 --trans_ckpt ./models/pretrained_v9.pth
python -W ignore train.py -n slerp_v9 -t Slerp -c 4 --max_epoch 2000 --skip_val_epoch 2000 --html_iter 50 --trans_ckpt ./models/pretrained_v9.pth
python -W ignore train.py -n basic_v5 -t Basic -c 4 --max_epoch 2000 --skip_val_epoch 2000 --html_iter 50 --trans_ckpt ./models/pretrained_v5.pth
python -W ignore train.py -n slerp_v5 -t Slerp -c 4 --max_epoch 2000 --skip_val_epoch 2000 --html_iter 50 --trans_ckpt ./models/pretrained_v5.pth