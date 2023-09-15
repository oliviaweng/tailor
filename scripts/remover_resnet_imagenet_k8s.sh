#!/bin/bash

num_layers=$1
experiment_name=$2
modifier=remove
batch_size=256
teacher_model_name=resnet${num_layers}_imagenet
student_model_name=resnet${num_layers}_imagenet
experiment=${student_model_name}_${modifier}_${experiment_name}

echo "teacher_name:"${teacher_model_name}
echo "student_name:"${student_model_name}
code_dir=/imagenet-volume/pytorch/python/src/imagenet_resnet
save_dir=/imagenet-volume/imagenet-training/pytorch_lightning
data_dir=/imagenet-volume/ILSVRC/Data/CLS-LOC
seed=1

pretrained_model_ckpt=${save_dir}/resnet50_imagenet_teacher_more/resnet50_imagenet_teacher_more_104-0.96.ckpt

resume_ckpt=/imagenet-volume/imagenet-training/pytorch_lightning/resnet50_imagenet_remove_resume/last.ckpt

python3 ${code_dir}/trainer_main.py \
   --data_dir ${data_dir} \
   --batch_size ${batch_size} \
   --num_workers 24 \
   --save_dir ${save_dir} \
   --experiment_name ${experiment} \
   --teacher_model ${teacher_model_name} \
   --student_model ${student_model_name} \
   --kd \
   --pretrained_model_ckpt ${pretrained_model_ckpt} \
   --modifier ${modifier} \
   --modifier_how_often 3 \
   --find_unused_parameters \
   --pin_memory \
   --checkpoint_resume_path ${resume_ckpt} \
