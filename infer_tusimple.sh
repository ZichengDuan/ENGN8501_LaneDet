#!/bin/bash
echo dataroot: $1

python test.py --dataroot $1 --name summer2winter_yosemite_pretrained --model test --no_dropout --tusimple --num_test 6666


mv results/summer2winter_yosemite_pretrained/test_latest/images/style/clips/* $1
