import torch as nn

from .blocks import *


class Classifier(nn.Module):

    def __init__(self, in_ch, k_way):
        super(Classifier, self).__init__()

        self.conv1 = input_stem(in_ch, 16)

        self.conv2 = stage_stem(16, 32)
        self.conv3 = stage_depthwise_separable_conv(32)
        self.conv4 = stage_stem(32, 64)
        self.conv5 = stage_depthwise_separable_conv(64)
        self.conv6 = stage_stem(64, 128)
        self.conv7 = stage_depthwise_separable_conv(128)
        self.conv8 = stage_stem(128, 256)
        self.conv9 = stage_depthwise_separable_conv(256)
        self.conv10 = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv10 = stage_stem(256, 512)
        # self.conv11 = stage_depthwise_separable_conv(512)
        self.logits = nn.Linear(256, k_way)
        

        # self.conv1 = ConvBlock(in_ch, 64)
        # self.conv2 = ConvBlock(64, 64)
        # self.conv3 = ConvBlock(64, 64)
        # self.conv4 = ConvBlock(64, 64)
        # self.conv1 = DepthwiseSeparableConvBlock(in_ch, 6, 12, pooling=True)
        # self.conv2 = DepthwiseSeparableConvBlock(12, 24, 48, pooling=True)
        # self.conv3 = DepthwiseSeparableConvBlock(48, 96, 48, pooling=True)
        # self.conv4 = DepthwiseSeparableConvBlock(48, 96, 48, pooling=True)
        # self.logits = nn.Linear(64*5*5, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # x = self.conv10(x)
        # x = self.conv11(x)
        x = x.view(x.shape[0], -1)
        x = self.logits(x)
        return x

        # x = self.conv1(x)
        #
        # identity = x
        # x = self.conv2(x)
        # # x += identity
        #
        # identity = x
        # x = self.conv3(x)
        # # x += identity
        #
        # identity = x
        # x = self.conv4(x)
        # # x += identity
        #
        # # print(x.shape)
        # x = x.view(x.shape[0], -1)
        # x = self.logits(x)
        # return x

    def functional_forward(self, x, params):
        """
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: model parameters,
                i.e. weights and biases of convolution
                     and weights and biases of
                                   batch normalization
                type is an OrderedDict

        Arguments:
        x: input images [batch, 1, 28, 28]
        params: The model parameters,
                i.e. weights and biases of convolution
                     and batch normalization layers
                It's an `OrderedDict`
        """
        x = input_stem_function(x, params, 1)
        x = stage_stem_function(x, params, 2)
        x = stage_depthwise_separable_conv_function(x, params, 3)
        x = stage_stem_function(x, params, 4)
        x = stage_depthwise_separable_conv_function(x, params, 5)
        x = stage_stem_function(x, params, 6)
        x = stage_depthwise_separable_conv_function(x, params, 7)
        x = stage_stem_function(x, params, 8)
        x = stage_depthwise_separable_conv_function(x, params, 9)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = stage_stem_function(x, params, 10)
        # x = stage_depthwise_separable_conv_function(x, params, 11)
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params["logits.weight"], params["logits.bias"])
        return x

        # for block in [1, 2, 3, 4]:
        #     x = ConvBlockFunction(
        #         x,
        #         params[f"conv{block}.0.weight"],
        #         params[f"conv{block}.0.bias"],
        #         params.get(f"conv{block}.1.weight"),
        #         params.get(f"conv{block}.1.bias"),
        #     )
        #     print(block, x.shape)
        #     if block != 1:
        #         identity = x
        #     pooling = True
        #     x = DepthwiseSeparableConvBlockFunction(
        #         x,
        #         params,
        #         block,
        #         pooling=pooling
        #     )
        #     # if block != 1:
        #     #     x += identity
        # x = x.view(x.shape[0], -1)
        # x = F.linear(x, params["logits.weight"], params["logits.bias"])
        # return x
