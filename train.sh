#function train {
    #config=local_configs/ddrformer/23slim/ddrformer.23slim.512x512.ade.160k.py
    #config=local_configs/ddrformer/23slim/ddrformer.23slim.1024x1024.city.160k.py
    #config=local_configs/ddrformer/23/ddrformer.23.1024x1024.city.160k.py
    #config=local_configs/ddrformer/23slim/ddrformer.23slim.512x1024.city.120k_4e_4wd0025min_lr0.py
    #config=local_configs/ddrformer/23slim/ddrformer.23slim.1024x1024.city.160k.py
    #python main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet18_tusimple_aug1b.py
#}
#export -f train
#export CUDA_VISIBLE_DEVICES=0
#,1,2,3
#nohup bash -c train > logs/resnet18_tusimple_aug1b.log &
#nohup python main_landet.py --train --config=configs/lane_detection/bezierlanenet/torch_1_7_resnet18_tusimple_aug1b_notf32.py > logs/torch_1_7_resnet18_tusimple_aug1b_notf32.log 2>&1 &
#nohup python main_landet.py --train python main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet18_llamas_aug1b.py --mixed-precision
export CUDA_VISIBLE_DEVICES=0
nohup python  main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet18_llamas_aug1b_large_kernel_attn.py --mixed-precision > logs/resnet18_llamas_aug1b_large_kernel_attn.log 2>&1 &
#nohup python main_landet.py --train --config=configs/lane_detection/scnn/resnet18_tusimple.py > logs/scnn_resnet18_tusimple_aug1b.log 2>&1 &
## multi-gpu training
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#tools/dist_train.sh \
#    local_configs/ddrformer/23slim/ddrformer.23slim.1024x1024.city.160k.py \
#    4 2>&1


## multi-gpu training
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#tools/dist_train.sh \
#    local_configs/ddrformer/23slim/ddrformer.23slim.1024x1024.city.160k.py \
#    4 2>&1
