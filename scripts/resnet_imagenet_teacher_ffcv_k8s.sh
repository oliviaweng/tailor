#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=None
batch_size=256
teacher_model_name=resnet${num_layers}_imagenet
student_model_name=None
experiment=${teacher_model_name}_${modifier}_${experiment_name}
echo "teacher_name:"${teacher_model_name}
echo "student_name:"${student_model_name}
code_dir=/imagenet-ffcv/pytorch/python/src/imagenet_resnet
save_dir=/imagenet-ffcv/imagenet-training/pytorch_lightning/
# data_dir=/imagenet-volume/ILSVRC/Data/CLS-LOC/
data_dir=/imagenet-ffcv


python3 ${code_dir}/trainer_main.py \
   --data_dir ${data_dir} \
   --batch_size ${batch_size} \
   --num_workers 24 \
   --save_dir ${save_dir} \
   --experiment_name ${experiment} \
   --teacher_model ${teacher_model_name} \
   --ffcv \
   --cosine_annealing \
