import math
import torch
import torch.nn.functional as F
from model.fresnetv2 import resnet18


class ArcLossMargin(torch.nn.Module):
    def __init__(self, input_dim, class_number, s=64, m=0.50, easy_margin=False):
        super(ArcLossMargin, self).__init__()
        self.input_dim = input_dim
        self.class_number = class_number
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(class_number, input_dim))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label, device='cpu', mixed_precision=False):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if mixed_precision:
            phi = phi.half()
        if self.easy_margin:
            # torch.where(condition, x, y) → Tensor
            # condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
            # x (Tensor) – values selected at indices where condition is True
            # y (Tensor) – values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine>0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将cos(\theta + m)更新到tensor相应的位置中
        one_hot = torch.zeros(cosine.size()).to(device)
        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        return output


def main():
    feature_dim = 512
    num_classes = 10000
    backbone = resnet18(num_classes=feature_dim, use_se=False)
    margin = ArcLossMargin(input_dim=feature_dim, class_number=num_classes)
    feature = backbone(torch.randn(5, 3, 112, 112))
    label = torch.LongTensor(5).random_(0, num_classes)
    output = margin(feature, label)
    print(output.size())
    pass


if __name__ == '__main__':
    main()
