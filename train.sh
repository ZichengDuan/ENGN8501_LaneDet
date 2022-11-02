export CUDA_VISIBLE_DEVICES=0
nohup python  main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet18_culane_AdamW_aug1b.py > logs/resnet18_culane_AdamW_aug1b.log 2>&1 &









