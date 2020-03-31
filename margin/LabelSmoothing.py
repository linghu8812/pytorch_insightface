import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    '''
    Implement label smoothing.  size表示类别总数
    '''

    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.LogSoftmax = nn.LogSoftmax()
        self.confidence = 1.0 - smoothing  # if i=y的公式
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """
        num_classes = x.size(-1)
        x = self.LogSoftmax(x)
        true_dist = x.data.clone()  # 先深复制过来
        true_dist.fill_(self.smoothing / (num_classes - 1))  # otherwise的公式
        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return self.criterion(x, true_dist)


if __name__ == '__main__':
    criterion = LabelSmoothing(smoothing=0.1)
    # predict.shape 3 5
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.9, 0.2, 0.1, 0],
                                 [1, 0.2, 0.7, 0.1, 0]])
    v = criterion(predict, torch.LongTensor([2, 1, 0]))
    print(v)
