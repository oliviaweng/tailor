#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=remove
teacher_model_name=resnet${num_layers}_imagenet
student_model_name=resnet${num_layers}_imagenet
experiment=${student_model_name}_${modifier}_${experiment_name}
pretrained_model_ckpt=/app/pytorch/python/src/imagenet_resnet/checkpoints/resnet50_imagenet_teacher_more_104-0.96.ckpt

resume_ckpt=/app/pytorch/python/src/imagenet_resnet/checkpoints/resnet50_imagenet_remove_wd1e-4_epoch37.ckpt

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
   --student_model ${student_model_name} \
   --kd \
   --pretrained_model_ckpt ${pretrained_model_ckpt} \
   --modifier ${modifier} \
   --modifier_how_often 3 \
   --find_unused_parameters \
   --checkpoint_resume_path ${resume_ckpt} \
