from resnet_imagenet import resnet18_imagenet, resnet34_imagenet, resnet50_imagenet, noskip_resnet50_imagenet, short_resnet50_imagenet, short_resnet50_teacher_imagenet, rd_noskip_resnet50_imagenet
from resnet_cifar10 import resnet8_cifar10, noskip_resnet8_cifar10, short_resnet8_cifar10_teacher, short_resnet8_cifar10

# Models that can be trained on their own
models = {
    'resnet50_imagenet': resnet50_imagenet,
    'resnet34_imagenet': resnet34_imagenet,
    'resnet18_imagenet': resnet18_imagenet,
    'noskip_resnet50_imagenet': noskip_resnet50_imagenet,
    'short_resnet50_imagenet': short_resnet50_imagenet,
    'short_resnet50_teacher_imagenet': short_resnet50_teacher_imagenet,
    'rd_noskip_resnet50_imagenet': rd_noskip_resnet50_imagenet,
    'resnet8_cifar10': resnet8_cifar10,
    'noskip_resnet8_cifar10': noskip_resnet8_cifar10,
    'short_resnet8_cifar10_teacher': short_resnet8_cifar10_teacher,
    'short_resnet8_cifar10': short_resnet8_cifar10,
}
