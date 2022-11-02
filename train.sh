export CUDA_VISIBLE_DEVICES=1
nohup python  main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet18_llamas_aug1b.py > logs/resnet18_llamas_aug1b.log 2>&1 &




