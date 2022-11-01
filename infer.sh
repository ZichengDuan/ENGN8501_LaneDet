#!/bin/bash
echo tusimple_abs_root: $1

python test.py --dataroot "suibian" --name summer2winter_yosemite_pretrained --model test --no_dropout --tusimple --tusimpleABSroot $1 --num_test 1000000 --preprocess none


mv results/summer2winter_yosemite_pretrained/test_latest/images/style/clips/* $1
