"""
Defines an unblur face model.

Test code path: test/unblur_face/unblur_face_test.py

Copyright 2024 Hideyuki Inada.  All rights reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mha import MultiHeadSelfAttention


class ResnetSingleBlock(nn.Module):
    """
    A single block for ResNet with optional skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        num_groups (int, optional): Number of groups for GroupNorm. Default is 32.
    """
    def __init__(self, in_channels, out_channels, stride=1, num_groups=32):
        super(ResnetSingleBlock, self).__init__()

        self.conv_skip = None
        if stride == 2 or in_channels != out_channels:
            self.conv_skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        effective_num_groups1 = min(num_groups, out_channels)
        effective_num_groups2 = min(num_groups, out_channels)

        self.gn1 = nn.GroupNorm(effective_num_groups1, out_channels)
        self.gn2 = nn.GroupNorm(effective_num_groups2, out_channels)

        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for conv modules.
        """
        if self.conv_skip is not None:
            nn.init.xavier_uniform_(self.conv_skip.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, inputs):
        """
        Forward pass for the ResNet single block.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        skip = inputs
        if self.conv_skip is not None:
            skip = self.conv_skip(skip)

        x = self.conv1(inputs)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = x + skip
        x = self.act2(x)
        return x


class ResnetSingleTransposeBlock(nn.Module):
    """
    A single transpose block for ResNet with optional skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        num_groups (int, optional): Number of groups for GroupNorm. Default is 32.
    """
    def __init__(self, in_channels, out_channels, stride=1, num_groups=32):
        super(ResnetSingleTransposeBlock, self).__init__()

        self.conv_skip = None
        if in_channels > out_channels:
            self.conv_skip = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                bias=False,
                padding=1
            )

            self.conv1 = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                bias=False,
                padding=1
            )
        else:
            self.conv1 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        effective_num_groups1 = min(num_groups, out_channels)
        effective_num_groups2 = min(num_groups, out_channels)

        if out_channels == 112:  # FIXME. Figure out the better way when 112%32 != 0
            effective_num_groups1 = 28
            effective_num_groups2 = 28
            
        self.gn1 = nn.GroupNorm(effective_num_groups1, out_channels)
        self.gn2 = nn.GroupNorm(effective_num_groups2, out_channels)

        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for conv modules.
        """
        if self.conv_skip is not None:
            nn.init.xavier_uniform_(self.conv_skip.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, inputs):
        """
        Forward pass for the ResNet single transpose block.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        skip = inputs
        if self.conv_skip is not None:
            skip = self.conv_skip(skip)

        x = self.conv1(inputs)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)

        x = x + skip
        x = self.act2(x)
        return x


class ResnetBlock(nn.Module):
    """
    A block consisting of multiple ResNet single blocks and optional multi-head self-attention.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        num_blocks (int, optional): Number of single blocks. Default is 6.
        heads (int, optional): Number of attention heads. Default is 8.
    """
    def __init__(self, in_channels, out_channels, stride=1, num_blocks=6, heads=8):
        super(ResnetBlock, self).__init__()

        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.blocks.append(ResnetSingleBlock(in_channels, out_channels, stride=stride))

        for _ in range(num_blocks - 1):
            self.blocks.append(
                ResnetSingleBlock(out_channels, out_channels, stride=1)
            )
        # Add 1 attention for each Resnet block
        if heads > 0:
            self.attentions.append(MultiHeadSelfAttention(out_channels, heads))

    def forward(self, inputs):
        """
        Forward pass for the ResNet block.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = inputs
        for block in self.blocks:
            x = block(x)

        if len(self.attentions) > 0:
            batch_size, channels, height, width = x.size()            
            x = x.view(batch_size, channels, -1).transpose(1, 2)  # Optimized reshape and permute
            x = self.attentions[0](x, x, x)
            x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Optimized reshape and permute back

        return x

class ResnetTransposeBlock(nn.Module):
    """
    A block consisting of multiple ResNet single transpose blocks and optional multi-head self-attention.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        num_blocks (int, optional): Number of single blocks. Default is 6.
        heads (int, optional): Number of attention heads. Default is 8.
    """
    def __init__(self, in_channels, out_channels, stride=1, num_blocks=6, heads=8):
        super(ResnetTransposeBlock, self).__init__()

        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.blocks.append(ResnetSingleTransposeBlock(in_channels, out_channels, stride=stride))

        for _ in range(num_blocks - 1):
            self.blocks.append(ResnetSingleTransposeBlock(out_channels, out_channels, stride=1))
        if heads > 0:
            self.attentions.append(MultiHeadSelfAttention(out_channels, heads))

    def forward(self, inputs):
        """
        Forward pass for the ResNet transpose block.

        ArgsContinuing from the previous part, here's the rest of the implementation with added docstrings for each method and argument:
        Forward pass for the ResNet transpose block.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = inputs
        for block in self.blocks:
            x = block(x)

        if len(self.attentions) > 0:
            batch_size, channels, height, width = x.size()
            x = x.view(batch_size, channels, -1).transpose(1, 2)  # Optimized reshape and permute
            x = self.attentions[0](x, x, x)
            x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Optimized reshape and permute back
        return x


class ConvAct(torch.nn.Module):
    """
    A convolutional layer followed by a SiLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Padding added to all four sides of the input. Default is 0.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 **kwargs):
        super().__init__(**kwargs)

        self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
        )

        self.act = torch.nn.SiLU()
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize a conv module weight.
        """
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, inputs):
        """
        Forward pass for the ConvAct block.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(inputs)
        outputs = self.act(x)
        return outputs


class ConvTransposeAct(torch.nn.Module):
    """
    A transpose convolutional layer followed by a SiLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Padding added to all four sides of the input. Default is 0.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 **kwargs):
        super().__init__(**kwargs)

        self.conv = torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
        )

        self.act = torch.nn.SiLU()

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the conv module weight.
        """
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, inputs):
        """
        Forward pass for the ConvTransposeAct block.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(inputs)
        outputs = self.act(x)
        return outputs


class UnblurCremageModelV6(torch.nn.Module):
    """
    Unblur face model implementation.

    Args:
        **kwargs: Additional arguments for the model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.down_blocks = torch.nn.ModuleList()
        self.mid_blocks = torch.nn.ModuleList()
        self.up_blocks = torch.nn.ModuleList()

        # module_list = list()
        # h,w: 256 128 64 32 16    8  4     2    1
        # channels
        #        3  16 32 64 128 256 512 1024 2048
        # strides  2  2  2   2  2   2   2    1
        #   k      3  3  3   3  3   3   3    2
        # pad:     1  1  1   1  1   1   1    0
        #   h:       0  1  2   3   4
        # h channel:16 32 64 128 256 512 1024 2048
        #   i:       0  1  2   3   4

        num_down_blocks = 8

        ns =  [3, 16, 32, 64, 128, 256, 512, 1024]
        ns2 = [16,32, 64, 128, 256, 512, 1024, 2048]
        ks =        [ 3,  3,  3,  3, 3, 3, 3, 2]
        paddings =  [ 1,  1,  1,  1, 1, 1, 1, 0]
        strides = [2, 2, 2, 2, 2, 2, 2, 1]
        heads = [0, 0, 0, 8, 8, 8, 8, 0]  # last one is ignored as k != 3
        for i in range(num_down_blocks):
            if ks[i] == 3:
                conv = ResnetBlock(ns[i], ns2[i], strides[i], heads=heads[i])
            else:
                conv = ConvAct(ns[i], ns2[i], ks[i], strides[i], padding=paddings[i])
            self.down_blocks.append(conv)

        self.mid_blocks.append(ConvAct(2048, 4096, 1))
        self.mid_blocks.append(MultiHeadSelfAttention(4096, heads=8))
        self.mid_blocks.append(ConvAct(4096, 2048, 1))

        num_up_blocks = num_down_blocks + 1
        # heigh, width      1 -  > 2  -> 4   -> 8  -> 16 -> 32 -> 642 -> 128 -> 256
        # channel        2048     1024   512   256    128   64    32     16       3
        # h rev          2048     1024   512   256    128   64    32     16
        # skip first
        # with concat
        #  h (w/o ignore)   0     1024   512   256    128   64    32     16       0  0
        #  c + prev      2048     1024  1024   768    512  320   192    112      64  3
        # in             2048     2048  1536  1024    640  384   224    128      64
        # out : see the cell diagonally up to the right
        # h is only skipped at the bottom layer (as it is the same output the midlayer has)
        strides = [2, 2, 2, 2, 2, 2, 2, 2, 1]
        ns =  [2048, 2048, 1536, 1024,  640, 384, 224, 128, 64]
        ns2 = [1024, 1024,  768,  512,  320, 192, 112,  64,  3]
        ks =    [4, 4, 4, 4, 4, 4, 4, 4, 3]  # use k=4, s=2, pad=1 for normal up
        heads = [0, 8, 8, 8, 8, 0, 0, 0, 0]
        for i in range(num_up_blocks):
            padding = 1
            if i < num_up_blocks - 1:
                if ks[i] == 4 and i > 0:
                    conv = ResnetTransposeBlock(ns[i], ns2[i], strides[i], heads=heads[i])
                else:
                    conv = ConvTransposeAct(ns[i], ns2[i], ks[i], strides[i], padding=padding)
            else: # last block, no activation
                conv = torch.nn.Conv2d(ns[i], ns2[i], ks[i], strides[i], padding=1)
            self.up_blocks.append(conv)

    def forward(self, inputs):
        """
        Forward pass for the UnblurCremageModelV6.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = inputs
        h = list()
        for block in self.down_blocks:
            x = block(x)
            h.append(x)

        for i, block in enumerate(self.mid_blocks):
            if isinstance(block, MultiHeadSelfAttention):
                batch_size, channels, height, width = x.size()

                x = x.view(batch_size, channels, -1).transpose(1, 2)  # Optimized reshape and permute
                x = block(x, x, x)
                x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Optimized reshape and permute back

            else:
                x = block(x)

        h.reverse()
        for i, block in enumerate(self.up_blocks):
            if i > 0 and i < len(self.up_blocks)-1:
                x = torch.concat([x, h[i]], dim=1)
            x = block(x)

        return x
