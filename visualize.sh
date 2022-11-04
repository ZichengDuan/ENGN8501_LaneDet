python tools/vis/lane_img_dir.py --config=configs/lane_detection/bezierlanenet/vis_resnet18_tusimple_aug1b.py --metric tusimple --style line

python tools/vis/lane_img_dir.py --config=/root/workspace/code/origin_code/pytorch-auto-drive/configs/lane_detection/scnn/vis_resnet18_scnn.py --metric tusimple --style line --checkpoint=/root/workspace/code/origin_code/pytorch-auto-drive/checkpoints/SCNN/resnet18_scnn_tusimple_20210424.pt


python main_landet.py --test --config=/root/workspace/code/origin_code/pytorch-auto-drive/configs/lane_detection/scnn/resnet18_tusimple.py

# culane label
python tools/vis/lane_img_dir.py --image-path=/root/workspace/data/culane/driver_37_30frame/05190751_0363.MP4 --keypoint-path=/root/workspace/data/culane/driver_37_30frame/05190751_0363.MP4 --image-suffix=.jpg --keypoint-suffix=.lines.txt --save-path=/root/workspace/code/ENGN8501_LaneDet/culane_vis --config=/root/workspace/code/ENGN8501_LaneDet/configs/lane_detection/bezierlanenet/resnet18_culane_aug1b.py