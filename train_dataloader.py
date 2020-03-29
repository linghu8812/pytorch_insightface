import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import time

train_dataset_transform = transforms.Compose([
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


class FaceRecognitionTrainDataset(Dataset):
    """Face Recognition Train dataset."""

    def __init__(self, train_list, transform=None, cache_images=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cache_images = cache_images
        self.image_labels = []
        with open(train_list, 'r') as f:
            self.image_list = f.read().splitlines()
        if self.cache_images:
            self.image_caches = []
            start_time = time.time()
            for info in self.image_list:
                _, image_name, image_label = info.split('\t')
                if int(image_label) > 999:
                    break
                print(f'Reading {image_name} ...', end="\r")
                src_img = cv2.imread(image_name)
                src_img = src_img[:, :, ::-1].copy()
                self.image_caches.append(src_img)
                self.image_labels.append(int(image_label))
            print(f'Reading has taken {time.time() - start_time} s')
        else:
            self.image_names = []
            start_time = time.time()
            for info in self.image_list:
                _, image_name, image_label = info.split('\t')
                if int(image_label) > 999:
                    break
                print(f'Reading {image_name} ...', end="\r")
                self.image_names.append(image_name)
                self.image_labels.append(int(image_label))
            print(f'Reading has taken {time.time() - start_time} s')

        self.class_nums = len(np.unique(self.image_labels))
        self.transform = transform

    def __len__(self):
        if self.cache_images:
            return len(self.image_caches)
        else:
            return len(self.image_names)

    def __getitem__(self, idx):
        # random flip with ratio of 0.5
        if self.cache_images:
            image = self.image_caches[idx]
        else:
            image = Image.open(self.image_names[idx])

        if self.transform is not None:
            image = self.transform(image)
        # else:
        #     image = transforms.ToTensor(image)

        label = torch.from_numpy(np.array(self.image_labels[idx]))
        return image, label


if __name__ == '__main__':
    face_dataset = FaceRecognitionTrainDataset(train_list='./dataset/train.lst', transform=train_dataset_transform)
    train_loader = DataLoader(face_dataset, batch_size=8, shuffle=True, num_workers=4)
    image, label = iter(train_loader).next()
    sample = image[0].squeeze()
    sample = sample.permute((1, 2, 0)).numpy()
    sample *= [0.5, 0.5, 0.5]
    sample += [0.5, 0.5, 0.5]
    plt.imshow(sample)
    plt.show()
    print('Label is: {}'.format(label[0].numpy()))
