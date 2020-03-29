import torch
from torchsummary import summary


def get_list(filter_number, units, num_classes):
    filter_list = [[filter_number[0]]]
    for index, unit in enumerate(units):
        layer_filters = [filter_number[index]]
        for i in range(unit):
            layer_filters.append(filter_number[index + 1])
            layer_filters.append(filter_number[index + 1])
        filter_list.append(layer_filters)
    filter_list.append([num_classes])
    return filter_list


class BasicBlock(torch.nn.Module):
    def __init__(self, layer_number, stride=1, use_se=False):
        super(BasicBlock, self).__init__()
        self.dim_match =True
        if stride == 2:
            self.dim_match = False
        self.use_se = use_se
        self.main_path = torch.nn.Sequential(
            torch.nn.BatchNorm2d(layer_number[0], eps=2e-5, momentum=0.9),
            torch.nn.Conv2d(layer_number[0], layer_number[1], kernel_size=3,
                            stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(layer_number[1], eps=2e-5, momentum=0.9),
            torch.nn.PReLU(num_parameters=layer_number[1]),
            torch.nn.Conv2d(layer_number[1], layer_number[2], kernel_size=3,
                            stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(layer_number[1], eps=2e-5, momentum=0.9)
        )
        self.res_path = torch.nn.Sequential(
            torch.nn.Conv2d(layer_number[0], layer_number[2], kernel_size=1,
                            stride=stride, padding=0, bias=False),
            torch.nn.BatchNorm2d(layer_number[1], eps=2e-5, momentum=0.9)
        )
        self.se_path = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Conv2d(layer_number[2], layer_number[2] // 16, kernel_size=1,
                            stride=1, padding=0, bias=False),
            torch.nn.PReLU(num_parameters=layer_number[2] // 16),
            torch.nn.Conv2d(layer_number[2] // 16, layer_number[2], kernel_size=1,
                            stride=stride, padding=0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        if not self.dim_match:
            res = self.res_path(x)
        else:
            res = x
        out = self.main_path(x)
        if self.use_se:
            x = self.se_path(out)
            out = out * x
        out = out + res
        return out


class ResNet(torch.nn.Module):
    def __init__(self, filter_list, use_se=False):
        super(ResNet, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, filter_list[0][0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(filter_list[0][0], eps=2e-5, momentum=0.9)
        self.relu0 = torch.nn.PReLU(num_parameters=filter_list[0][0])
        self.stage1 = self.residual_unit(BasicBlock, filter_list[1], use_se)
        self.stage2 = self.residual_unit(BasicBlock, filter_list[2], use_se)
        self.stage3 = self.residual_unit(BasicBlock, filter_list[3], use_se)
        self.stage4 = self.residual_unit(BasicBlock, filter_list[4], use_se)
        self.output_layer = torch.nn.Sequential(torch.nn.BatchNorm2d(filter_list[-2][-1]),
                                                torch.nn.Dropout(0.4),
                                                torch.nn.Flatten(),
                                                torch.nn.Linear(filter_list[-2][-1] * 49, filter_list[-1][0]),
                                                torch.nn.BatchNorm1d(filter_list[-1][0]))

    def residual_unit(self, block, filter_list, use_se=False):
        layers = []
        for i in range(0, len(filter_list), 2):
            layer_number = filter_list[i:i+3]
            if len(layer_number) == 3:
                if i == 0:
                    layers.append(block(layer_number, stride=2, use_se=use_se))
                else:
                    layers.append(block(layer_number, stride=1, use_se=use_se))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.output_layer(x)
        return x


def resnet18(num_classes=512, use_se=False):
    filter_number = [64, 64, 128, 256, 512]
    units = [2, 2, 2, 2]
    filter_list = get_list(filter_number, units, num_classes)
    return ResNet(filter_list, use_se)


def resnet100(num_classes=512, use_se=False):
    filter_number = [64, 64, 128, 256, 512]
    units = [3, 13, 30, 3]
    filter_list = get_list(filter_number, units, num_classes)
    return ResNet(filter_list, use_se)


def main():
    net = resnet18(use_se=True)
    summary(net, (3, 112, 112), device="cpu")
    y = net(torch.randn(5, 3, 112, 112))
    print(y.size())


if __name__ == '__main__':
    main()
