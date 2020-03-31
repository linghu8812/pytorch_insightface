import matplotlib.pyplot as plt
import numpy as np
import torch
from train_dataloader import FaceRecognitionTrainDataset, DataLoader, train_dataset_transform


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    return data, targets, shuffled_targets, lam


def visual_mixup(input_image, input_label, alpha):
    mix_image, label_a, label_b, lam = mixup_data(input_image, input_label, alpha)
    sample = mix_image[0].squeeze()
    sample = sample.permute((1, 2, 0)).numpy()
    sample *= [0.5, 0.5, 0.5]
    sample += [0.5, 0.5, 0.5]
    plt.imshow(sample)
    plt.show()
    print(f'Label a is: {label_a[0].numpy()}, label b is: {label_b[0].numpy()}')


if __name__ == '__main__':
    face_dataset = FaceRecognitionTrainDataset(train_list='./dataset/train2.lst', transform=train_dataset_transform)
    train_loader = DataLoader(face_dataset, batch_size=8, shuffle=True, num_workers=4)
    image, label = iter(train_loader).next()
    visual_mixup(image, label, 1.5)
    visual_mixup(image, label, 1.0)
    visual_mixup(image, label, 0.5)
