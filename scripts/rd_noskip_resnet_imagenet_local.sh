#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=none
weight_decay=1e-5
teacher_model_name=rd_noskip_resnet${num_layers}_imagenet
student_model_name=None
experiment=${teacher_model_name}_${modifier}_${experiment_name}
echo "teacher_name:"${teacher_model_name}
echo "student_name:"${student_model_name}
save_dir=/app/pytorch/imagenet-training/${experiment}/
data_dir=/app/pytorch/small-imagenet
seed=1

python3 trainer_main.py \
   --data_dir ${data_dir} \
   --batch_size 1 \
   --save_dir ${save_dir} \
   --experiment_name ${experiment} \
   --teacher_model ${teacher_model_name} \
   --fast_dev_run \
                       
                       
                       
                  
