#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=remove
weight_decay=1e-4
teacher_model_name=resnet${num_layers}_cifar10
student_model_name=resnet${num_layers}_cifar10
aim=pl_${teacher_model_name}_${modifier}
experiment=${teacher_model_name}_${modifier}_${experiment_name}
echo "teacher_name:"${teacher_model_name}
echo "student_name:"${student_model_name}
save_dir=/app/pytorch/cifar10-training/

pretrained_model_ckpt="/app/pytorch/cifar10-training/pl_resnet8_cifar10_teacher/teacher/teacher_epoch=148_loss=0.37_top1=0.87.ckpt"

python3 trainer_main.py \
   --batch_size 32 \
   --num_workers 24 \
   --save_dir ${save_dir} \
   --experiment_name ${experiment_name} \
   --teacher_model ${teacher_model_name} \
   --student_model ${student_model_name} \
   --kd \
   --pretrained_model_ckpt ${pretrained_model_ckpt} \
   --modifier ${modifier} \
   --modifier_how_often 3 \
   --find_unused_parameters \
   --dataset cifar10 \
   --num_epochs 200
                       
                       
                       
                  
