import torch
from model.fresnetv2 import resnet18


class SoftmaxMargin(torch.nn.Module):
    def __init__(self, input_dim, class_number):
        super(SoftmaxMargin, self).__init__()
        self.softmax_margin = torch.nn.Linear(input_dim, class_number)

    def forward(self, x):
        out = self.softmax_margin(x)
        return out


def main():
    feature_dim = 512
    num_classes = 10000
    backbone = resnet18(num_classes=feature_dim, use_se=False)
    margin = SoftmaxMargin(input_dim=feature_dim, class_number=num_classes)
    feature = backbone(torch.randn(5, 3, 112, 112))
    output = margin(feature)
    print(output.size())


if __name__ == '__main__':
    main()
