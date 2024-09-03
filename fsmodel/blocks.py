import torch.nn as nn
import torch.nn.functional as F


def input_stem(in_ch: int, out_ch: int = 16):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6()
    )


def input_stem_function(x, params, block: int = 1):
    x = F.conv2d(x, params[f"conv{block}.0.weight"], params[f"conv{block}.0.bias"], stride=2, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=params[f"conv{block}.1.weight"],
                     bias=params[f"conv{block}.1.bias"], training=True)
    x = F.relu6(x)
    return x


def stage_stem(in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(),
    )


def stage_stem_function(x, params, block: int):
    x = F.conv2d(x, params[f"conv{block}.0.weight"], params[f"conv{block}.0.bias"], stride=2, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=params[f"conv{block}.1.weight"],
                     bias=params[f"conv{block}.1.bias"], training=True)
    x = F.relu6(x)
    return x


def stage_depthwise_separable_conv(ch: int, stride: int = 1):
    return nn.Sequential(
        nn.Conv2d(ch, ch * 6, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(ch * 6),
        nn.ReLU6(),
        nn.Conv2d(ch * 6, ch * 6, kernel_size=3, stride=stride, padding=1, groups=ch * 6, bias=False),
        nn.BatchNorm2d(ch * 6),
        nn.ReLU6(),
        nn.Conv2d(ch * 6, ch, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(ch),
        nn.ReLU6(),
    )


def stage_depthwise_separable_conv_function(x, params, block: int, stride: int = 1):
    x = F.conv2d(x, params[f"conv{block}.0.weight"], params[f"conv{block}.0.bias"])
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=params[f"conv{block}.1.weight"],
                     bias=params[f"conv{block}.1.bias"], training=True)
    x = F.relu6(x)

    x = F.conv2d(x, params[f"conv{block}.3.weight"], None, stride=stride, padding=1, groups=x.size(1))
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=params[f"conv{block}.4.weight"],
                     bias=params[f"conv{block}.4.bias"], training=True)
    x = F.relu6(x)

    x = F.conv2d(x, params[f"conv{block}.6.weight"], params[f"conv{block}.6.bias"])
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=params[f"conv{block}.7.weight"],
                     bias=params[f"conv{block}.7.bias"], training=True)
    x = F.relu6(x)
    return x


def ConvBlock(in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(
        x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True
    )
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


def DepthwiseSeparableConvBlock(in_ch: int, mid_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                                padding: int = 1, pooling: bool = False):
    layers = [
        # Pointwise convolution
        nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU6(inplace=True),

        # Depthwise convolution
        nn.Conv2d(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=mid_ch, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU6(inplace=True),

        # Pointwise convolution to get to the desired out_ch
        nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_ch),
        # nn.ReLU6(inplace=True),
    ]

    if pooling:
        layers.append(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    return nn.Sequential(*layers)


def DepthwiseSeparableConvBlockFunction(x, params, block: int, pooling: bool = False):
    # Pointwise convolution
    x = F.conv2d(x, params[f"conv{block}.0.weight"], params[f"conv{block}.0.bias"])
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=params[f"conv{block}.1.weight"],
                     bias=params[f"conv{block}.1.bias"], training=True)
    x = F.relu6(x, inplace=True)

    # Depthwise convolution
    x = F.conv2d(x, params[f"conv{block}.3.weight"], None, padding=1, groups=x.size(1))
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=params[f"conv{block}.4.weight"],
                     bias=params[f"conv{block}.4.bias"], training=True)
    x = F.relu6(x, inplace=True)

    # Pointwise convolution
    x = F.conv2d(x, params[f"conv{block}.6.weight"], params[f"conv{block}.6.bias"])
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=params[f"conv{block}.7.weight"],
                     bias=params[f"conv{block}.7.bias"], training=True)

    if pooling:
        x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


def conv_1x1_bn(in_channels, out_channels, act_func):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        get_activation_layer(act_func)
    )


def get_activation_layer(act_func):
    if act_func == 'relu6':
        return nn.ReLU6(inplace=True)
    else:
        return nn.ReLU(inplace=True)


class MBInvertedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, mid_channels, act_func, use_se):
        super(MBInvertedConvLayer, self).__init__()
        self.use_se = use_se
        self.act_func = act_func
        self.expand_ratio = expand_ratio

        self.expanded_channels = in_channels * expand_ratio
        self.mid_channels = mid_channels

        # Expansion layer
        self.expansion_conv = None
        if self.expand_ratio != 1:
            self.expansion_conv = conv_1x1_bn(in_channels, self.expanded_channels, act_func)

        # Depthwise convolution layer
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.expanded_channels, self.expanded_channels, kernel_size, stride, kernel_size // 2,
                      groups=self.expanded_channels, bias=False),
            nn.BatchNorm2d(self.expanded_channels),
            get_activation_layer(act_func)
        )

        # Projection layer
        self.projection_conv = conv_1x1_bn(self.expanded_channels, out_channels, act_func)

    def forward(self, x):
        if self.expansion_conv:
            x = self.expansion_conv(x)
        x = self.depthwise_conv(x)
        x = self.projection_conv(x)
        return x


class MobileInvertedResidualBlock(nn.Module):
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()
        self.conv = MBInvertedConvLayer(**mobile_inverted_conv)
        self.has_shortcut = shortcut is not None
        if self.has_shortcut:
            self.shortcut = IdentityLayer(**shortcut)

    def forward(self, x):
        if self.has_shortcut:
            return self.shortcut(x) + self.conv(x)
        else:
            return self.conv(x)


class IdentityLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order):
        super(IdentityLayer, self).__init__()
        # Identity layer does nothing if in_channels == out_channels
        # Here we assume that is the case. Otherwise, more logic is needed to match dimensions.
        pass

    def forward(self, x):
        return x


# Create the specified blocks
# block1 = MobileInvertedResidualBlock({
#     "in_channels": 8,
#     "out_channels": 16,
#     "kernel_size": 5,
#     "stride": 2,
#     "expand_ratio": 6,
#     "mid_channels": 48,
#     "act_func": "relu6",
#     "use_se": False
# }, None)
#
# block2 = MobileInvertedResidualBlock({
#     "in_channels": 16,
#     "out_channels": 16,
#     "kernel_size": 3,
#     "stride": 1,
#     "expand_ratio": 4,
#     "mid_channels": 64,
#     "act_func": "relu6",
#     "use_se": False
# }, {
#     "in_channels": 16,
#     "out_channels": 16,
#     "use_bn": False,
#     "act_func": None,
#     "dropout_rate": 0,
#     "ops_order": "weight_bn_act"
# })
