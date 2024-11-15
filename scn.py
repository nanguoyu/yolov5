import torch
import torch.nn as nn
import torchvision.models as models
import copy
from torch import nn
import torch
from typing import List
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from typing import Union, List, Dict, Any
from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)


class ScnLayer(nn.Module):
    def __init__(self, original_layer, d) -> None:
        super(ScnLayer, self).__init__()
        self.original_layer = copy.deepcopy(original_layer)
        self.original_layer.requires_grad_(False)
        self.d = d
        self.device = next(original_layer.parameters()).device
        self.has_bias = self.original_layer.bias is not None
        self.creat_d_weight_list()

    def creat_d_weight_list(self):
        # for linear and conv2d modules, there are only weight and bias.
        # for attention-like layers in LLMs, you should implement this function in subclass
        self.weight_list = nn.ParameterList([nn.Parameter(self.original_layer.weight.clone().detach().data) for _ in range(self.d)])
        if self.has_bias:
            self.bias_list = nn.ParameterList([nn.Parameter(self.original_layer.bias.clone().detach().data) for _ in range(self.d)])

    def configuration(self, beta):
        assert self.d == len(beta), "inconsistent dimensions"
        # for linear and conv2d modules, there are only weight and bias.
        # for attention-like layers, you should implement this function in subclass
        weight_list = [a*b for a, b in zip(self.weight_list, beta)]
        self.weight = torch.sum(torch.stack(weight_list), dim=0)
        if self.has_bias:
            bias_list = [a*b for a, b in zip(self.bias_list, beta)]
            self.bias = torch.sum(torch.stack(bias_list), dim=0)
        else:
            self.bias = None

    def forward(self, x):
        raise NotImplementedError("You should implement this in sub-class module.")

class ScnLinear(ScnLayer):
    def __init__(self, original_layer, d):
        super(ScnLinear, self).__init__(original_layer, d)
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

    def forward(self, x):
        # print("scn_linear forward")
        return nn.functional.linear(x, self.weight, self.bias)

class ScnConv2d(ScnLayer):
    def __init__(self, original_layer, d):
        super(ScnConv2d, self).__init__(original_layer, d)
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups

    def forward(self, x):
        # print("scn_conv2d forward")
        return nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ScnConv1d(ScnLayer):
    def __init__(self, original_layer, d):
        super(ScnConv1d, self).__init__(original_layer, d)
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups

    def forward(self, x):
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

class ScnYolov5Conv(nn.Module):
    # Conv in yolov5
    def __init__(self, original_layer:Conv, d, sub=False):
        super(ScnYolov5Conv, self).__init__()
        self.conv  = ScnConv2d(original_layer.conv, d)
        self.bn = original_layer.bn
        self.act = original_layer.act
        if not sub:
            self.i = original_layer.i
            self.f = original_layer.f
            self.type = original_layer.type
            self.np = original_layer.np

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
    def configuration(self, beta):
        self.conv.configuration(beta)

class ScnYolov5Bottleneck(nn.Module):
    # Bottleneck in yolov5
    def __init__(self, original_layer:Bottleneck, d, sub=False):
        super(ScnYolov5Bottleneck, self).__init__()
        self.cv1 = ScnYolov5Conv(original_layer.cv1, d, sub=True)
        self.cv2 = ScnYolov5Conv(original_layer.cv2, d, sub=True)
        self.add = original_layer.add
        if not sub:
            self.i = original_layer.i
            self.f = original_layer.f
            self.type = original_layer.type
            self.np = original_layer.np

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
    def configuration(self, beta):
        self.cv1.configuration(beta)
        self.cv2.configuration(beta)

class ScnYolov5C3(nn.Module):
    # C3 in yolov5
    def __init__(self, original_layer:C3, d):
        super(ScnYolov5C3, self).__init__()
        self.cv1 = ScnYolov5Conv(original_layer.cv1, d, sub=True)
        self.cv2 = ScnYolov5Conv(original_layer.cv2, d, sub=True)
        self.cv3 = ScnYolov5Conv(original_layer.cv3, d, sub=True)
        self.n = len(original_layer.m)
        self.m = nn.Sequential(*[ScnYolov5Bottleneck(layer, d, sub=True) for layer in original_layer.m])
        self.i = original_layer.i
        self.f = original_layer.f
        self.type = original_layer.type
        self.np = original_layer.np
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    
    def configuration(self, beta):
        self.cv1.configuration(beta)
        self.cv2.configuration(beta)
        self.cv3.configuration(beta)
        for layer in self.m:
            layer.configuration(beta)


def replace_layers(model, d, config:dict):
    print("Replace layers...........")
    modules_to_replace = []
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            full_child_name = f"{name}.{child_name}" if name != "" else child_name
            print(f"check {full_child_name}")
            if full_child_name in config["layers"] and config["layers"][full_child_name]:
                if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear, C3, Bottleneck, Conv)):
                    print("yes")
                    modules_to_replace.append((module, name, child_name, child_module))

    for parent_module, parent_name, child_name, child_module in modules_to_replace:
        full_child_name = f"{parent_name}.{child_name}" if parent_name != "" else child_name
        if isinstance(child_module, nn.Linear):
            setattr(parent_module, child_name, ScnLinear(child_module, d))
            print(f"Replaced {full_child_name} with ScnLinear")
        elif isinstance(child_module, nn.Conv2d):
            setattr(parent_module, child_name, ScnConv2d(child_module, d))
            print(f"Replaced {full_child_name} with ScnConv2d")
        elif isinstance(child_module, nn.Conv1d):
            setattr(parent_module, child_name, ScnConv1d(child_module, d))
            print(f"Replaced {full_child_name} with ScnConv1d")
        elif isinstance(child_module, C3):
            setattr(parent_module, child_name, ScnYolov5C3(child_module, d))
            print(f"Replaced {full_child_name} with ScnYolov5C3")
        elif isinstance(child_module, Bottleneck):
            setattr(parent_module, child_name, ScnYolov5Bottleneck(child_module, d))
            print(f"Replaced {full_child_name} with ScnYolov5Bottleneck")
        elif isinstance(child_module, Conv):
            setattr(parent_module, child_name, ScnYolov5Conv(child_module, d))
            print(f"Replaced {full_child_name} with ScnYolov5Conv")

class SCN(nn.Module):
    def __init__(self, num_alpha:int, dimensions:int, base_model:nn.Module, config:dict=None) -> None:
        super(SCN, self).__init__()
        print(f"Create SCN(D={dimensions}, num_alpha={num_alpha}).......")
        self.dimensions=dimensions
        self.num_alpha = num_alpha
        self.hyper_stack  = nn.Sequential(
            nn.Linear(self.num_alpha, 64),
            nn.ReLU(),
            nn.Linear(64, self.dimensions),
            nn.Softmax(dim=0)
        )

        self.base_model_template = copy.deepcopy(base_model)

        self.base_model = base_model
        # todo: you can add a SCN_parameter_names to control which layers you want to apply SCN
        # and then only apply replace_layers to those layers
        replace_layers(self.base_model, self.dimensions, config)

    def __getitem__(self, idx):
        if isinstance(self.base_model, nn.Sequential):
            return self.base_model[idx]
        raise TypeError("Base model is not subscriptable")

    def __len__(self):
        if isinstance(self.base_model, nn.Sequential):
            return len(self.base_model)
        raise TypeError("Base model is not subscriptable")
    
    def hyper_forward_and_configure(self, hyper_x):
        hyperout = self.hyper_stack(hyper_x)
        for name, module in self.base_model.named_modules():
            if isinstance(module, ScnLayer):
                module.configuration(hyperout)

    def forward(self, x):
        return self.base_model(x)

    def export_hypernet(self) -> nn.Module:
        """
        Export the hypernetwork part, including hyper_stack
        Returns:
            nn.Module: A new instance of hypernetwork with copied parameters
        """
        # Create a new hypernetwork with the same structure
        new_hypernet = nn.Sequential(
            nn.Linear(self.num_alpha, 64),
            nn.ReLU(),
            nn.Linear(64, self.dimensions),
            nn.Softmax(dim=0)
        )

        # Copy parameters
        with torch.no_grad():
            for new_param, old_param in zip(new_hypernet.parameters(), self.hyper_stack.parameters()):
                new_param.data = old_param.data.cpu().detach().clone()

        return new_hypernet

    def export_basemodel(self, i: int) -> nn.Module:
        """
        Export the i-th base model
        Args:
            i: Index of the base model to export (0 to dimensions-1)
        Returns:
            nn.Module: A new instance of base model with the i-th weight set
        Raises:
            ValueError: If index i is out of range
        """
        if not 0 <= i < self.dimensions:
            raise ValueError(f"Index {i} out of range [0, {self.dimensions-1}]")

        # Create new model from template
        exported_model = copy.deepcopy(self.base_model_template)

        # Collect all SCN layer weights from current model
        scn_state_dict = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, ScnLayer):
                # Record the i-th weight set of current SCN layer
                weight_data = module.weight_list[i].data.cpu().detach()
                if module.has_bias:
                    bias_data = module.bias_list[i].data.cpu().detach()
                else:
                    bias_data = None
                scn_state_dict[name] = (weight_data, bias_data)

        # Set weights to corresponding layers in exported model
        for name, module in exported_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                if name in scn_state_dict:
                    weight_data, bias_data = scn_state_dict[name]
                    with torch.no_grad():
                        module.weight.data.copy_(weight_data)
                        if bias_data is not None:
                            module.bias.data.copy_(bias_data)

        return exported_model