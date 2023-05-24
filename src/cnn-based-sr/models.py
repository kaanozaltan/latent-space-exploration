import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
    def get_name(self):
        return 'srcnn'


class VDSR(nn.Module):
    def __init__(self, num_channels=3, num_filters=64, scaling_factor=0.1, num_convs=18):
        super(VDSR, self).__init__()
        self.conv_in = nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1)
        self.convs = nn.ModuleList([self._build_conv(num_filters) for _ in range(num_convs)])
        self.conv_out = nn.Conv2d(num_filters, num_channels, kernel_size=3, padding=1)
        self.scaling_factor = scaling_factor

    def _build_conv(self, num_filters):
        return nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        # x_in = x
        x = self.conv_in(x)

        for block in self.convs:
            x = block(x)

        # x = self.conv_out(residual + x_in)
        x = self.conv_out(x)
        return x
    
    def get_name(self):
        return 'vdsr'


class EDSR(nn.Module):
    def __init__(self, num_channels=3, num_filters=64, scaling_factor=0.1, num_residuals=20):
        super(EDSR, self).__init__()
        self.conv_in = nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1)
        self.residuals = nn.ModuleList([self._build_residual(num_filters) for _ in range(num_residuals)])
        self.conv_out = nn.Conv2d(num_filters, num_channels, kernel_size=3, padding=1)
        self.scaling_factor = scaling_factor

    def _build_residual(self, num_filters):
        return nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv_in(x)
        x_in = x

        for block in self.residuals:
            residual = x
            residual = block(residual)
            x = x + residual

        x = self.conv_out(x + x_in)
        return x
    
    def get_name(self):
        return 'edsr'
