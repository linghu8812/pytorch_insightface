import os
import torch
from test_dataloader import FaceRecognitionTestDataset, DataLoader, val_dataset_transform
from model.fresnetv2 import resnet100
from sklearn import preprocessing
import numpy as np
import cv2
import time
import tqdm


def get_features(loader, len_loader, len_data):
    features = []
    processing_bar = tqdm.tqdm(enumerate(loader), total=len_loader)
    start_time = time.time()
    for index, data in processing_bar:
        data = data.to(device)
        embeddings = backbone(data)
        embeddings = embeddings.cpu().numpy()
        embeddings = preprocessing.normalize(embeddings)
        features.extend(embeddings)
    end_time = time.time()
    print(f'Averaging processing time is {(end_time - start_time) / len_data}s ...')
    features = np.array(features)
    return features


batch_size = 512
feature_dim = 512
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data ...')
anchor_dataset = FaceRecognitionTestDataset(image_path=b'/media/F/face_test_dataset/alignment_imgs/anchor_imgs',
                                            transform=val_dataset_transform, b_anchor=True)
anchor_loader = DataLoader(anchor_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)
len_anchor = len(anchor_loader)

test_dataset = FaceRecognitionTestDataset(image_path='/media/F/face_test_dataset/alignment_imgs/test_imgs',
                                          transform=val_dataset_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)
len_test = len(test_loader)

print('==> Building model ...')
backbone = resnet100(num_classes=feature_dim)
if os.path.exists('weights/backbone.pth'):
    backbone.load_state_dict(torch.load('weights/backbone.pth')['net'])
backbone = backbone.to(device)
backbone.eval()

print('==> Processing data ...')
with torch.no_grad():
    anchor_features = get_features(anchor_loader, len_anchor, len(anchor_dataset))
    test_features = get_features(test_loader, len_test, len(test_dataset))

    similarity = anchor_features.dot(test_features.transpose())
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for i in range(similarity.shape[1]):
        top1 = np.argmax(similarity[:, i])
        name = anchor_dataset.image_labels[top1]
        prob = np.round((1.0 / (1 + np.exp(-9.8 * similarity[top1, i] + 3.763)) * 1.0023900077282935 +
                         -1.290327970702342e-06), decimals=4) * 100
        # if prob < 60:
        #     continue
        prob = f'{prob:.2f}'
        src_img = cv2.imread(test_dataset.image_names[i])
        rst_name = os.path.join(result_dir, name + '-' + prob + '-' + test_dataset.image_names[i].split('.')[0]
                                .split('/')[-1] + '.jpg')
        cv2.imwrite(rst_name, src_img)
        print('Saving {} ...'.format(rst_name))
