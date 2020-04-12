import os
import torch
from test_dataloader import FaceRecognitionTestDataset, DataLoader, val_dataset_transform
from model.fresnetv2 import resnet100

batch_size = 256
feature_dim = 512
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data ...')
anchor_dataset = FaceRecognitionTestDataset(image_path=b'/media/F/face_test_dataset/alignment_imgs/anchor_imgs',
                                            transform=val_dataset_transform, b_anchor=True)
anchor_loader = DataLoader(anchor_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)
len_train = len(anchor_loader)

test_dataset = FaceRecognitionTestDataset(image_path='/media/F/face_test_dataset/alignment_imgs/special_face_test',
                                          transform=val_dataset_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)
len_train = len(test_loader)

print('==> Building model ...')
backbone = resnet100(num_classes=feature_dim)
if os.path.exists('weights/backbone.pth'):
    backbone = torch.load('weights/backbone.pth')['net']
backbone = backbone.to(device)
backbone.eval()

for inputs in anchor_loader:
    inputs = inputs.to(device)
    feature = backbone(inputs)
    a = 0



