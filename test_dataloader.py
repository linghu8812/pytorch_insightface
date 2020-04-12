import os
from torch.utils.data import Dataset, DataLoader
from val_dataloader import val_dataset_transform
import matplotlib.pyplot as plt
from PIL import Image


class FaceRecognitionTestDataset(Dataset):
    """ Face  Recognition Train dataset."""

    def __init__(self, image_path, transform=None, flip=False, b_anchor=False):
        image_list = os.listdir(image_path)
        self.b_anchor = b_anchor
        self.image_names = []
        self.image_labels = []
        for image_name in image_list:
            if self.b_anchor:
                image_label = image_name.decode('utf-8')
                image_label = image_label.split('.')[0]
                self.image_labels.append(image_label)
            image_name = os.path.join(image_path, image_name)
            self.image_names.append(image_name)
        self.transform = transform
        self.flip = flip

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = Image.open(self.image_names[idx])
        if self.flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == '__main__':
    face_dataset = FaceRecognitionTestDataset(image_path=b'/media/F/face_test_dataset/alignment_imgs/anchor_imgs',
                                              transform=val_dataset_transform, flip=False, b_anchor=True)
    test_loader = DataLoader(face_dataset, batch_size=8, shuffle=False, num_workers=4)
    image = iter(test_loader).next()
    sample = image[0].squeeze()
    sample = sample.permute((1, 2, 0)).numpy()
    sample *= [0.5, 0.5, 0.5]
    sample += [0.5, 0.5, 0.5]
    plt.imshow(sample)
    plt.show()
