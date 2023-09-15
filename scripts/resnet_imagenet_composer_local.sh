#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=teacher
weight_decay=1e-4
model=resnet${num_layers}_imagenet
teacher_model_name=None
echo "teacher_name:"${teacher_model_name}
echo "student_name:"${model}
save_dir=/app/pytorch/imagenet-training/${experiment_name}/
data_dir=/app/pytorch/small-imagenet
seed=1

composer -n 1 tailor_trainer.py \
   --model ${model} \
   --data_dir ${data_dir} \
   --batch_size 1 \
   --save_dir ${save_dir} \
   --experiment_name ${experiment_name} \
                       
                       
                       
                  
