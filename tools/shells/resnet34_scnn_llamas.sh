#!/bin/bash
# Trained weights: resnet34_scnn_llamas_20210625.pt
# Training
python main_landet.py --train --config=configs/lane_detection/scnn/resnet34_llamas.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --val --config=configs/lane_detection/scnn/resnet34_llamas.py --mixed-precision
# Testing with official scripts
./autotest_llamas.sh resnet34_scnn_llamas val checkpoints
# Predict lane points for the eval server, find results in ./output
python main_landet.py --test --config=configs/lane_detection/scnn/resnet34_llamas.py --mixed-precision
