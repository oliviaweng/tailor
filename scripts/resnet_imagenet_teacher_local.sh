#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=teacher
weight_decay=1e-4
teacher_model_name=resnet${num_layers}_imagenet
student_model_name=None
aim=pl_${teacher_model_name}_${modifier}
echo "teacher_name:"${teacher_model_name}
echo "student_name:"${student_model_name}
save_dir=/app/pytorch/imagenet-training/${aim}/
data_dir=/app/pytorch/small-imagenet
seed=1

python3 trainer_main.py \
   --data_dir ${data_dir} \
   --batch_size 1 \
   --save_dir ${save_dir} \
   --experiment_name ${experiment_name} \
   --teacher_model ${teacher_model_name} \
   --fast_dev_run \
                       
                       
                       
                  
