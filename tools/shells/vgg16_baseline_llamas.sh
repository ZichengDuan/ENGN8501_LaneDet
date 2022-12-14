#!/bin/bash
# Trained weights: vgg16_baseline_llamas_20210625.pt
# Training
python main_landet.py --train --config=configs/lane_detection/baseline/vgg16_llamas.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --val --config=configs/lane_detection/baseline/vgg16_llamas.py --mixed-precision
# Testing with official scripts
./autotest_llamas.sh vgg16_baseline_llamas val checkpoints
# Predict lane points for the eval server, find results in ./output
python main_landet.py --test --config=configs/lane_detection/baseline/vgg16_llamas.py --mixed-precision
