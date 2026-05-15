"""
Visual base model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # initial conv
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        # global avg pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # initial conv and pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50_backbone(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        try:
            import torchvision.models as models
            pretrained_model = models.resnet50(pretrained=True)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() 
                             if k in model_dict and 'fc' not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        except ImportError:
            print("Warning: torchvision not available, skipping pretrained load")
        except Exception as e:
            print(f"Warning: failed to load pretrained weights: {e}")
    
    return model

def resnet50_backbone_6ch(pretrained=False, **kwargs):
    adapter = nn.Sequential(
        nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True)
    )

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    class ResNetWithAdapter(nn.Module):
        def __init__(self, adapter, resnet):
            super().__init__()
            self.adapter = adapter
            self.resnet = resnet
        def forward(self, x):
            return self.resnet(self.adapter(x))

    return ResNetWithAdapter(adapter, model)

def resnet50_backbone_12ch(pretrained=False, **kwargs):
    adapter = nn.Sequential(
        nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True)
    )

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    class ResNetWithAdapter(nn.Module):
        def __init__(self, adapter, resnet):
            super().__init__()
            self.adapter = adapter
            self.resnet = resnet
        def forward(self, x):
            return self.resnet(self.adapter(x))

    return ResNetWithAdapter(adapter, model)


class ExpertMLP(nn.Module):

    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * hidden_mult
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  
        return self.drop(self.fc2(self.act(self.fc1(x))))  


class ConditionalTopKMoE(nn.Module):

    def __init__(
        self,
        d_model: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        renorm_topk: bool = True,
        cond_mode: str = "mean",  
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.renorm_topk = renorm_topk
        self.cond_mode = cond_mode

        self.router_x = nn.Linear(d_model, num_experts)   
        self.router_c = nn.Linear(d_model, num_experts)   

        self.experts = nn.ModuleList(
            [ExpertMLP(d_model, hidden_mult=hidden_mult, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, F_main, F_cond):

        B, N, D = F_main.shape
        assert D == self.d_model
        assert F_cond.dim() == 3 and F_cond.size(-1) == D

        logits = self.router_x(F_main) + self.router_c(F_cond)   
        WG = F.softmax(logits, dim=-1)                            

        topk_w, topk_idx = WG.topk(self.top_k, dim=-1)            

        if self.renorm_topk:
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)

        Y = torch.stack([exp(F_main) for exp in self.experts], dim=2)  

        idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)         
        Y_topk = torch.gather(Y, dim=2, index=idx)                 

        X_moe = (topk_w.unsqueeze(-1) * Y_topk).sum(dim=2)         

        F_refined = F_main + X_moe                                  

        return X_moe, F_refined, WG


class BidirectionalMoE(nn.Module):

    def __init__(
        self,
        d_model: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        renorm_topk: bool = True,
    ):
        super().__init__()
        
        # syndrome -> organ MoE
        self.moe_syndrome_to_organ = ConditionalTopKMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            hidden_mult=hidden_mult,
            dropout=dropout,
            renorm_topk=renorm_topk,
        )
        
        # organ -> syndrome MoE
        self.moe_organ_to_syndrome = ConditionalTopKMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            hidden_mult=hidden_mult,
            dropout=dropout,
            renorm_topk=renorm_topk,
        )

    def forward(self, F_organ, F_syndrome):
        X_moe_s2o, F_organ_refined, WG_s2o = self.moe_syndrome_to_organ(
            F_main=F_organ,      
            F_cond=F_syndrome   
        )
        
        X_moe_o2s, F_syndrome_refined, WG_o2s = self.moe_organ_to_syndrome(
            F_main=F_syndrome,   
            F_cond=F_organ       
        )
        
        return (
            X_moe_s2o, F_organ_refined, WG_s2o,
            X_moe_o2s, F_syndrome_refined, WG_o2s
        )


# CycleTCM network
class CycleTCM(nn.Module):
    def __init__(self, num_classes1=8,num_classes2=5, pretrained=False, **kwargs):
        super(CycleTCM, self).__init__()
        self.backbone_whole = resnet50_backbone(pretrained=pretrained, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        embedding_dim = 2048

        self.bn = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(0.3)

        self.classifier1 = nn.Linear(embedding_dim, num_classes1)
        self.classifier2 = nn.Linear(embedding_dim, num_classes2)
        
    def forward(self, x_whole, x_edge, x_body, x_heart_lung, x_spleen, x_liver, x_kidney):
        
        x = self.backbone_whole(x_whole)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.dropout(x)
        syndrome_pred  = self.classifier1(x)
        organ_pred = self.classifier2(x)

        return syndrome_pred, organ_pred


# count model parameters
def count_parameters(model):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return total_params, trainable_params, non_trainable_params

# print model parameter summary
def print_model_info(model):
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    
    print("Model parameter summary")
    print(f"Total:        {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable:    {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Fixed:        {non_trainable_params:,} ({non_trainable_params/1e6:.2f}M)")


def _smoke_test():
    """Random-input forward pass; check shapes and parameter count (no ImageNet pretrained load)."""
    torch.manual_seed(0)
    model = CycleTCM(num_classes1=8, num_classes2=5, pretrained=False)
    model.eval()
    b, h, w = 2, 224, 224
    x_whole = torch.randn(b, 3, h, w)
    x_edge = torch.randn(b, 3, h, w)
    x_body = torch.randn(b, 3, h, w)
    x_heart_lung = torch.randn(b, 3, h, w)
    x_spleen = torch.randn(b, 3, h, w)
    x_liver = torch.randn(b, 3, h, w)
    x_kidney = torch.randn(b, 3, h, w)
    with torch.no_grad():
        syndrome_pred, organ_pred = model(
            x_whole, x_edge, x_body, x_heart_lung, x_spleen, x_liver, x_kidney
        )
    print("smoke_test ok")
    print("  syndrome_pred:", tuple(syndrome_pred.shape))
    print("  organ_pred:   ", tuple(organ_pred.shape))
    print_model_info(model)


if __name__ == "__main__":
    _smoke_test()

