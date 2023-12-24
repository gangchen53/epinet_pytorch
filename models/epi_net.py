import torch
import torch.nn as nn


class BasicLayer(nn.Module):
    """ Basic layer : depth x (Conv - ReLU - Conv - BN - ReLU) """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: int = 3,
                 ):
        super(BasicLayer, self).__init__()

        # (B, C, H, W) -> (B, C, H - 2 * depth, W - 2 * depth)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.basic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(depth - 1)
        ])

    def forward(self, inputs):
        x = self.conv_block(inputs)
        for layer in self.basic_layers:
            x = layer(x)
        outputs = x
        return outputs


class LastLayer(nn.Module):
    """ Last Layer : Conv - ReLU - Conv """
    def __init__(self, in_channels: int, out_channels: int):
        super(LastLayer, self).__init__()

        # (B, C, H, W) -> (B, C, H - 2, W - 2)
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=2, stride=1, padding=0)
        )

    def forward(self, inputs):
        outputs = self.last_layer(inputs)
        return outputs


class EPINet(nn.Module):
    # Note: if the shape of input features is (B, C, H, W), the shape of output features is (B, C, H - 22, W - 22).
    def __init__(self,
                 in_channels: int = 7,
                 multistream_layer_channels: int = 70,
                 multistream_layer_depth: int = 3,
                 merge_layer_depth: int = 7,
                 ):
        super(EPINet, self).__init__()

        self.multistream_layer = BasicLayer(
            in_channels=in_channels,
            out_channels=multistream_layer_channels,
            depth=multistream_layer_depth,
        )
        self.merge_layer = BasicLayer(
            in_channels=4 * multistream_layer_channels,
            out_channels=4 * multistream_layer_channels,
            depth=merge_layer_depth,
        )
        self.last_layer = LastLayer(in_channels=4 * multistream_layer_channels, out_channels=1)

    def forward(self, inputs):
        input_stack_90d, input_stack_0d, input_stack_45d, input_stack_m45d = inputs

        mid_90d = self.multistream_layer(input_stack_90d)
        mid_0d = self.multistream_layer(input_stack_0d)
        mid_45d = self.multistream_layer(input_stack_45d)
        mid_m45d = self.multistream_layer(input_stack_m45d)

        mid_merge = torch.cat([mid_90d, mid_0d, mid_45d, mid_m45d], dim=1)
        mid_merge = self.merge_layer(mid_merge)
        outputs = self.last_layer(mid_merge)
        return outputs
