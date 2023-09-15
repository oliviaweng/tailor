#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=None
batch_size=256
weight_decay=1e-4
model=resnet${num_layers}_imagenet
teacher_model_name=None

experiment=${model}_${modifier}_${experiment_name}
echo "teacher_name:"${teacher_model_name}
echo "student_name:"${model}
code_dir=/imagenet-ffcv/pytorch/python/src/imagenet_resnet
save_dir=/imagenet-ffcv/imagenet-training/composer
# data_dir=/imagenet-volume/ILSVRC/Data/CLS-LOC/
data_dir=/imagenet-ffcv

num_gpu=4

composer -n ${num_gpu} -v ${code_dir}/tailor_trainer.py \
   --model ${model} \
   --data_dir ${data_dir} \
   --batch_size ${batch_size} \
   --num_workers 8 \
   --save_dir ${save_dir} \
   --experiment_name ${experiment_name} \
   --pin_memory \
   --num_gpu ${num_gpu} \
   --ffcv \
