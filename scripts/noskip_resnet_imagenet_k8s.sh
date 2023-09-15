#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=none
batch_size=256
weight_decay=1e-5
teacher_model_name=noskip_resnet${num_layers}_imagenet
student_model_name=None
experiment=${teacher_model_name}_${modifier}_${experiment_name}
echo "teacher_name:"${teacher_model_name}
echo "student_name:"${student_model_name}
code_dir=/imagenet-volume/pytorch/python/src/imagenet_resnet
save_dir=/imagenet-volume/imagenet-training/pytorch_lightning/
data_dir=/imagenet-volume/ILSVRC/Data/CLS-LOC/
seed=1

python3 ${code_dir}/trainer_main.py \
   --data_dir ${data_dir} \
   --batch_size ${batch_size} \
   --num_workers 28 \
   --save_dir ${save_dir} \
   --experiment_name ${experiment} \
   --teacher_model ${teacher_model_name} \
                       
                       
                       
                  
