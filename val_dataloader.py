import cv2
import time
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

val_dataset_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in range(4):
        data = np.zeros((len(issame_list)*2, image_size[0], image_size[1], 3)).astype('uint8')
        data_list.append(data)

    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        image = cv2.imdecode(np.asarray(bytearray(_bin), dtype="uint8"), cv2.IMREAD_COLOR)
        image = image[:, :, ::-1].copy()
        for flip in [0, 1]:
            if flip == 1:
                image = cv2.flip(image, 1)
            data_list[flip][i][:] = image
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


class FaceRecognitionValDataset(Dataset):
    """ Face  Recognition Train dataset."""

    def __init__(self, bin_file, transform=None):
        image_size = (112, 112)
        self.dataset_name = bin_file.split('/')[-1].split('.')[0]
        print(f'Loading {self.dataset_name} ...')
        start_time = time.time()
        self.data_list, self.issame_list = load_bin(bin_file, image_size)
        print(f'Loading {self.dataset_name} cost {time.time() - start_time:.2f} sec')
        self.transform = transform

    def __len__(self):
        return len(self.issame_list)*2

    def __getitem__(self, idx):
        image0 = self.data_list[0][idx]
        image1 = self.data_list[1][idx]
        if self.transform is not None:
            image0 = self.transform(image0)
            image1 = self.transform(image1)
        return image0, image1


if __name__ == '__main__':
    face_dataset = FaceRecognitionValDataset(bin_file='./dataset/lfw.bin',
                                             transform=val_dataset_transform)
    val_loader = DataLoader(face_dataset, batch_size=8, shuffle=False, num_workers=4)
    image1, image2 = iter(val_loader).next()
    sample1, sample2 = image1[0].squeeze(), image2[0].squeeze()
    sample1, sample2 = sample1.permute((1, 2, 0)).numpy(), sample2.permute((1, 2, 0)).numpy()
    sample1, sample2 = sample1 * [0.5, 0.5, 0.5], sample2 * [0.5, 0.5, 0.5]
    sample1, sample2 = sample1 + [0.5, 0.5, 0.5], sample2 + [0.5, 0.5, 0.5]
    plt.imshow(sample1)
    plt.show()
    plt.imshow(sample2)
    plt.show()
