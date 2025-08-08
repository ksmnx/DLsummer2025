from torch import nn


class MFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, linear=False):
        super().__init__()
        if linear:
            self.layer = nn.Linear(in_channels, out_channels * 2)
        else:
            self.layer = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding)
        self.out_channels = out_channels
        self.linear = linear

    def forward(self, x):
        x = self.layer(x)
        out = torch.chunk(x, 2, dim=1)
        return torch.max(out[0], out[1])

class LCNN(nn.Module):
    def __init__(self, num_classes=2, input_shape=(1, 64, 600)):
        super().__init__()

        def mfm(in_c, mfm_out_c, k, s, pool=True):
            layers = [
                nn.Conv2d(in_c, mfm_out_c * 2, kernel_size=k, stride=s, padding=k[0] // 2),
                MFM(mfm_out_c * 2, mfm_out_c),
                nn.BatchNorm2d(mfm_out_c)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.layers = nn.Sequential(
            mfm(1, 48, (5, 5), (1, 1)),
            mfm(48, 96, (1, 1), (1, 1), pool=False),
            mfm(96, 128, (3, 3), (1, 1), pool=False),
        )

        with torch.no_grad():
            size = self.layers(torch.zeros(1, *input_shape)).view(1, -1).size(1)

        self.l1 = nn.Linear(size, 256)
        self.bn_l1 = nn.BatchNorm1d(256)
        self.mfm_l1 = MFM(256, 128, linear=True)
        self.bn_l2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.7)
        self.l2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.bn_l1(x)
        x = self.mfm_l1(x)
        x = self.bn_l2(x)
        x = self.dropout(x)
        return self.l2(x)

