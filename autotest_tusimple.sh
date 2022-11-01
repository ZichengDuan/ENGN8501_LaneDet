#!/bin/bash
###
 # @Author: Yunxiang Liu u7191378@anu.edu.au
 # @Date: 2022-10-14 16:02:43
 # @LastEditors: Yunxiang Liu u7191378@anu.edu.au
 # @LastEditTime: 2022-10-21 16:50:43
 # @FilePath: \pytorch-auto-drive\autotest_tusimple.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

data_dir=/root/datasets/tusimple/
echo experiment name: $1
echo status: $2
echo save dir: $3

# Perform test/validation with official scripts
cd tools/tusimple_evaluation
if [ "$2" = "test" ]; then
    python lane.py ../../output/${1}.json ${data_dir}test_label.json $1 $3
else
    python lane.py ../../output/${1}.json ${data_dir}label_data_0531.json $1 $3
fi
cd ../../
