import torch
import torch.nn as nn
import math



class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size=3, stride=1, use_se=True):
        super(GhostBottleneck, self).__init__()
        assert hidden_dim%2 == 0
        assert oup%2 == 0
        self.hdc = hidden_dim//2
        self.opc = oup//2
        self.use_se = use_se
        self.stride = stride
        self.opcut = False
        # ghost 1
        self.relu = nn.ReLU(inplace=True)
        self.conv1p = nn.Conv2d(inp, self.hdc, kernel_size=1, bias=False)
        self.bn1p = nn.BatchNorm2d(self.hdc)
        self.conv1c = nn.Conv2d(self.hdc, self.hdc, kernel_size=3, padding=1, groups=self.hdc, bias=False)
        self.bn1c = nn.BatchNorm2d(self.hdc)
        # stride
        if stride>=2:
            self.conv_s = nn.Conv2d(hidden_dim, hidden_dim, 
                                        kernel_size=kernel_size, padding=kernel_size//2, 
                                        stride=stride, groups=hidden_dim, bias=False)
            self.bn_s = nn.BatchNorm2d(hidden_dim)
        # se
        if self.use_se:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim//4)
            self.fc2 = nn.Linear(hidden_dim//4, hidden_dim)
        # ghost 2
        self.conv2p = nn.Conv2d(hidden_dim, self.opc, kernel_size=1, bias=False)
        self.bn2p = nn.BatchNorm2d(self.opc)
        self.conv2c = nn.Conv2d(self.opc, self.opc, kernel_size=3, padding=1, groups=self.opc, bias=False)
        self.bn2c = nn.BatchNorm2d(self.opc)
        # shotcut
        if stride!=1 or inp!=oup:
            self.conv_sc1 = nn.Conv2d(inp, inp, kernel_size=3, padding=1, stride=stride, groups=inp, bias=False)
            self.bn_sc1 = nn.BatchNorm2d(inp)
            self.conv_sc2 = nn.Conv2d(inp, oup, kernel_size=1, bias=False)
            self.bn_sc2 = nn.BatchNorm2d(oup)
            self.opcut = True
        
    def forward(self, x):
        # res
        if self.opcut:
            res = self.relu(self.bn_sc1(self.conv_sc1(x)))
            res = self.bn_sc2(self.conv_sc2(res))
        else:
            res = x
        # ghost 1
        x1 = self.relu(self.bn1p(self.conv1p(x)))
        x2 = self.relu(self.bn1c(self.conv1c(x1)))
        x = torch.cat([x1,x2], dim=1)
        # stride
        if self.stride>=2:
            x = self.relu(self.bn_s(self.conv_s(x)))
        # se
        if self.use_se:
            b, c, _, _ = x.shape
            factor = self.avg_pool(x).view(b, c)
            factor = self.fc2(self.relu(self.fc1(factor)))
            factor = torch.clamp(factor, 0, 1).view(b, -1, 1, 1)
            x = x*factor
        x1 = self.bn2p(self.conv2p(x))
        x2 = self.bn2c(self.conv2c(x1))
        x = torch.cat([x1,x2], dim=1)
        x = x + res
        return self.relu(x)



class GhostNet(nn.Module):
    def __init__(self, num_classes=1000, wd=1):
        super(GhostNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 16*wd, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16*wd)
        # block 12
        block12 = [
            GhostBottleneck(16*wd, 16*wd, 16*wd, kernel_size=3, stride=1, use_se=0),
            GhostBottleneck(16*wd, 48*wd, 24*wd, kernel_size=3, stride=2, use_se=0),
            GhostBottleneck(24*wd, 72*wd, 24*wd, kernel_size=3, stride=1, use_se=0)]
        self.block12 = nn.Sequential(*block12)
        # block 3
        block3 = [
            GhostBottleneck(24*wd, 72*wd, 40*wd, kernel_size=5, stride=2, use_se=1),
            GhostBottleneck(40*wd, 120*wd, 40*wd, kernel_size=5, stride=1, use_se=1)]
        self.block3 = nn.Sequential(*block3)
        # block 4
        block4 = [
            GhostBottleneck(40*wd, 240*wd, 80*wd, kernel_size=3, stride=2, use_se=0),
            GhostBottleneck(80*wd, 200*wd, 80*wd, kernel_size=3, stride=1, use_se=0),
            GhostBottleneck(80*wd, 184*wd, 80*wd, kernel_size=3, stride=1, use_se=0),
            GhostBottleneck(80*wd, 184*wd, 80*wd, kernel_size=3, stride=1, use_se=0),
            GhostBottleneck(80*wd, 480*wd, 112*wd, kernel_size=3, stride=1, use_se=1),
            GhostBottleneck(112*wd, 672*wd, 112*wd, kernel_size=3, stride=1, use_se=1)]
        self.block4 = nn.Sequential(*block4)
        # block 5
        block5 = [
            GhostBottleneck(112*wd, 672*wd, 160*wd, kernel_size=5, stride=2, use_se=1),
            GhostBottleneck(160*wd, 960*wd, 160*wd, kernel_size=5, stride=1, use_se=0),
            GhostBottleneck(160*wd, 960*wd, 160*wd, kernel_size=5, stride=1, use_se=1),
            GhostBottleneck(160*wd, 960*wd, 160*wd, kernel_size=5, stride=1, use_se=0),
            GhostBottleneck(160*wd, 960*wd, 160*wd, kernel_size=5, stride=1, use_se=1)]
        self.block5 = nn.Sequential(*block5)
        # fc
        self.out_fn = [40*wd, 112*wd, 160*wd]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(160*wd, num_classes)
        # init
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block12(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = GhostNet(wd=1)
    x = torch.rand(2,3,224,224)
    out = net(x)
    print(out.shape)
