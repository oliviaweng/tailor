#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=teacher
weight_decay=1e-4
teacher_model_name=noskip_resnet${num_layers}_cifar10
student_model_name=None
aim=pl_${teacher_model_name}_${modifier}
echo "teacher_name:"${teacher_model_name}
echo "student_name:"${student_model_name}
save_dir=/app/pytorch/cifar10-training/${aim}/

python3 trainer_main.py \
   --batch_size 32 \
   --num_workers 24 \
   --save_dir ${save_dir} \
   --experiment_name ${experiment_name} \
   --teacher_model ${teacher_model_name} \
   --find_unused_parameters \
   --dataset cifar10 \
   --num_epochs 200
                       
                       
                       
                  
