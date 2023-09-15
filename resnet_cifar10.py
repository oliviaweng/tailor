import math, torch, pdb
import torch.nn.functional as F
import torch.nn as nn
inplace = False
affine = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        x   = self.relu(x)
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class Bottleneckwoskip(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneckwoskip, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x, **kwargs):
        x   = self.relu(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, 
        inplanes, 
        planes, 
        stride=1, 
        downsample=None, 
        use_residual=True,
        **kwargs
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.downsample = downsample
        self.stride = stride
        self.use_residual = use_residual
        
    def forward(self, x, **kwargs):
        x   = self.relu(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None and self.use_residual:
            identity = self.downsample(x)

        if self.use_residual:
            out = out + identity

        return out

class ShortenableBasicBlock(nn.Module):
    """
    A Shortenable basic block. We convert the traditional long skip
    connection that spans the entire block into two shorter skip connections.
    This can be done on the fly during training.
    """
    expansion = 1

    def __init__(
        self, 
        inplanes, 
        planes, 
        stride=1, 
        downsample=None, 
        short_skip=False, 
        use_residual=True,
        **kwargs):
        super(ShortenableBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.downsample = downsample
        self.stride = stride
        self.short_skip = short_skip
        self.use_residual = use_residual
        
    def forward(self, x, **kwargs):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
    
        if self.short_skip:
            if self.downsample is not None:
                identity = self.downsample(identity)
            assert out.shape == identity.shape, 'first short skip mismatch'
            out = out + identity
            identity = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if not self.short_skip and self.use_residual:
            if self.downsample is not None:
                identity = self.downsample(x) 

        out = out + identity

        return out
    
class BasicBlockwoskip(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlockwoskip, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        x   = self.relu(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        return out   
          
class ResNet(nn.Module):
    def __init__(
        self, block, 
        blockwoskip, 
        layers_blocks, 
        blockchoice, 
        num_classes=100, 
        mul=1.0, 
        short_skip=False, 
        use_residual=True
    ):
        super(ResNet, self).__init__()
        self.mul = mul
        self.inplanes = int(16 * self.mul)
        self.layers = nn.ModuleList()
        self.blockchoice = blockchoice
        self.layer_blocks= layers_blocks
        self.block = block
        self.short_skip = short_skip
        self.use_residual = use_residual
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(16 * self.mul), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(16 * self.mul), affine=affine),
            nn.ReLU(inplace=True))

        self.layers.append(self._make_layer(block, blockwoskip, self.blockchoice[0], int(16*self.mul),  layers_blocks[0]          ))
        self.layers.append(self._make_layer(block, blockwoskip, self.blockchoice[1], int(32*self.mul), layers_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, blockwoskip, self.blockchoice[2], int(64*self.mul), layers_blocks[2], stride=2))
        # self.layers.append(self._make_layer(block, blockwoskip, self.blockchoice[3], int(512*self.mul), layers_blocks[3], stride=2))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(64*self.mul) * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        zero_init_residual = False
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, blockwoskip, blockchoice, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, affine=affine),
            )
        layers = []
        thisblock = block if blockchoice[0] == 1 else blockwoskip
        layers.append(
            thisblock(
                self.inplanes, 
                planes,
                stride, 
                downsample, 
                short_skip=self.short_skip,
                use_residual=self.use_residual
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            thisblock = block if blockchoice[i] == 1 else blockwoskip
            layers.append(
                thisblock(
                    self.inplanes, 
                    planes, 
                    short_skip=self.short_skip,
                    use_residual=self.use_residual
                )
            )
        return nn.Sequential(*layers)
    
    def get_residual_layer_indices(self):
        """
        Returns dictionary that maps a skip connection index to self.layers'
        tuple indices, since ResNet's residual layers are stored as a list of
        lists
        """
        idx = 0
        residual_layer_indices = {}
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                residual_layer_indices[idx] = (i, j)
                idx = idx + 1
        return residual_layer_indices

    def num_shortcuts(self):
        """
        Count the number of shortcut/skip connections are in the model
        """
        shortcut_count = 0
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                if self.layers[i][j].use_residual:
                    shortcut_count = shortcut_count + 1
        return shortcut_count

    def update_skip_removal(self, how_often, epoch):
        """
        Remove skip connection every how_often epochs, starting from the head of
        the network.

        Return True if skip removed; o.w. False
        """
        # [1, 1, 1] = 3 total skip connections
        # NOTE: Both epoch and skip connections are 0-indexed.
        skip_to_remove = epoch // how_often - 1 
        # ----------------------------------^^^ 
        # |- Subtract 1 to have first removal start at epoch how_often. This way
        # the model will train for how_often epochs first to establish some
        # optimizer states.
        num_shortcuts = self.num_shortcuts()
        print(f"Num skips left before removal = {num_shortcuts}")
        res_layer_indices = self.get_residual_layer_indices()
        if num_shortcuts > 0:
            for s in range(skip_to_remove + 1): # + 1 bc range is exclusive
                if s in res_layer_indices.keys():
                    print(f"Checking skip {s} gone at end of epoch {epoch}")
                    i = res_layer_indices[s][0]
                    j = res_layer_indices[s][1]
                    self.layers[i][j].use_residual = False
                    print(f"Num skips left = {self.num_shortcuts()}")
                else:
                    break

        if epoch != 0 and epoch % how_often == 0 \
        and skip_to_remove in res_layer_indices.keys():
            print(f"\n\nskip {skip_to_remove} REMOVED at end of epoch {epoch}")
            i = res_layer_indices[skip_to_remove][0]
            j = res_layer_indices[skip_to_remove][1]
            self.layers[i][j].use_residual = False
            print(f"Num skips left = {self.num_shortcuts()}")
            return True
        return False

    def update_skip_shorten(self, how_often, epoch):
        """
        Shorten skip connection every how_often epochs, starting from the head of
        the network.

        Return True if skip shortened; o.w. False 
        """
        # [1, 1, 1] = 3 total skip connections
        # NOTE: Both epoch and skip connections are 0-indexed.
        skip_to_shorten = epoch // how_often - 1 
        # ----------------------------------^^^ 
        # |- Subtract 1 to have first shorten start at epoch how_often. This way
        # the model will train for how_often epochs first to establish some
        # optimizer states.
        num_shortcuts = self.num_shortcuts()
        print(f"Num skips left before shortening = {num_shortcuts}")
        res_layer_indices = self.get_residual_layer_indices()
        if num_shortcuts > 0:
            for s in range(skip_to_shorten + 1): # + 1 bc range is exclusive
                if s in res_layer_indices.keys():
                    print(f"Checking skip {s} gone at end of epoch {epoch}")
                    i = res_layer_indices[s][0]
                    j = res_layer_indices[s][1]
                    self.layers[i][j].use_residual = False
                    self.layers[i][j].short_skip = True
                    print(f"Num skips left = {self.num_shortcuts()}")
                else:
                    break

        if epoch != 0 and epoch % how_often == 0 \
        and skip_to_shorten in res_layer_indices.keys():
            print(f"\n\nskip {skip_to_shorten} SHORTENED at end of epoch {epoch}")
            i = res_layer_indices[skip_to_shorten][0]
            j = res_layer_indices[skip_to_shorten][1]
            self.layers[i][j].use_residual = False
            self.layers[i][j].short_skip = True
            print(f"Num skips left = {self.num_shortcuts()}")
            return True
        return False

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                x = self.layers[i][j](x, **kwargs)
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward_towindow(self, x, **kwargs):
        dis_point= kwargs.get("dis_point")
        dis_fea  = []
        layer_index = 0
        
        x = self.conv1(x)
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                x = self.layers[i][j](x)
                if layer_index in dis_point:
                    dis_fea.append(x)
                layer_index += 1 
            if i == 1:
                x = x.detach()
                    
        dis_fea.append(x)
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, dis_fea
    
    def forward_bt(self, x, **kwargs):
        x = self.conv1(x)
        return x
    
    def forward_bl(self, x, **kwargs):
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward_to(self, x, **kwargs):
        dis_point = kwargs.get("dis_point")
        dis_fea  = []
        layer_index = 0
        x = self.conv1(x)
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                x = self.layers[i][j](x)
                if layer_index in dis_point:
                    dis_fea.append(x)
                layer_index += 1 
        dis_fea.append(x)
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, dis_fea
                
    def forward_from(self, x, **kwargs):
        se_index = kwargs.get("se_index")
        layer_index = 0
        dis_fea = []
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                if layer_index < se_index:
                    layer_index += 1
                else:
                    x = self.layers[i][j](x)
        dis_fea.append(x)      
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, dis_fea
    
    def get_bn_before_relu(self, dis_point):
        
        layer_index = 0
        bn = []
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                if layer_index in dis_point:
                    if isinstance(self.layers[i][j], BasicBlock):
                        bn.append(self.layers[i][j].bn2)
                    else:
                        bn.append(self.layers[i][j].bn3)
                layer_index += 1
        return bn
                
    def get_layer_blocks(self):
        return self.layer_blocks
    
    def get_blockchoice(self):
        return self.blockchoice
    
    def get_channel_num(self, dis_point):
        layer_index = 0
        channels = []
        channel = [[64 * 4]*3, [128* 4]*4, [256* 4]*6, [512* 4]*3]
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                if layer_index in dis_point:
                    channels.append(channel[i][j])
                layer_index += 1
        return channels
    
    def get_base_channel(self, dis_point):
        num = len(dis_point) + 1
        if self.block == Bottleneck:
            return [2048]*num
        elif self.block == BasicBlock:
            return [512]*num
    
#block, blockwoskip, layers, blockchoice, num_classes=1000
def resnet50(blockchoice, num_classes=100, mul=1.0):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet(Bottleneck, Bottleneckwoskip, [3, 4, 6, 3], blockchoice, num_classes, mul=mul)

def resnet34(blockchoice, num_classes=100, mul=1.0):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet(BasicBlock, BasicBlockwoskip, [3, 4, 6, 3], blockchoice, num_classes, mul=mul)

def resnet18(blockchoice, num_classes=100, mul=1.0):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet(BasicBlock, BasicBlockwoskip, [2, 2, 2, 2], blockchoice, num_classes, mul=mul)

def resnet8(blockchoice, num_classes=10, mul=1.0):
    r"""ResNet-8 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet(BasicBlock, BasicBlockwoskip, [1, 1, 1], blockchoice, num_classes, mul=mul)

def short_resnet8(blockchoice, num_classes=10, mul=1.0):
    """
    ResNet8 with shortened skip connections
    """
    return ResNet(
        ShortenableBasicBlock, 
        ShortenableBasicBlock, # NOTE: Only here to meet class init requirements
        [1, 1, 1], 
        blockchoice, 
        num_classes, 
        mul=mul,
        short_skip=True
    )

def short_resnet8_teacher(blockchoice, num_classes=10, mul=1.0):
    """
    ResNet8 built with ShortenableBasic blocks but use the default,
    conventional long skip connections (that span the whole Basic block)
    """
    return ResNet(
        ShortenableBasicBlock, 
        ShortenableBasicBlock, # NOTE: Only here to meet class init requirements
        [1, 1, 1], 
        blockchoice, 
        num_classes, 
        mul=mul,
        short_skip=False
    )

def resnet8_cifar10():
    return resnet8(blockchoice=[[1], [1], [1]])

def noskip_resnet8_cifar10():
    return resnet8(blockchoice=[[0], [0], [0]])

def short_resnet8_cifar10_teacher():
    return short_resnet8_teacher(blockchoice=[[0], [0], [0]])

def short_resnet8_cifar10():
    return short_resnet8(blockchoice=[[0], [0], [0]])

if __name__ == "__main__":
    blockchoice = [[0]*3, [1]*4, [1]*6, [1]*3]
    model = resnet50(blockchoice)
    print(model)