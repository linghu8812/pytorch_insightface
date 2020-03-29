import os
import time
import numpy as np
import math
import sklearn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model.fresnetv2 import resnet18
from margin.ArcLoss import ArcLossMargin
from train_dataloader import FaceRecognitionTrainDataset, train_dataset_transform, DataLoader
from val_dataloader import FaceRecognitionValDataset, val_dataset_transform
from verification import evaluate

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    mixed_precision = False
mixup = 1
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
learning_rate = 0.01
batch_size = 256
accumulate = 2
epochs = 50
feature_dim = 512
warmup = 4000
writer = SummaryWriter(log_dir='logs')

# Data
print('==> Preparing data ...')

train_dataset = FaceRecognitionTrainDataset(train_list='./dataset/train.lst', transform=train_dataset_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8)
len_train = len(train_loader)

val_dataset = FaceRecognitionValDataset(bin_file='../../data/faces_emore/lfw.bin',
                                        transform=val_dataset_transform)
len_lfw = len(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)
len_val = len(val_loader)

print('==> Building model ...')
backbone = resnet18(num_classes=feature_dim)
backbone = backbone.to(device)
margin = ArcLossMargin(input_dim=feature_dim, class_number=train_dataset.class_nums)
margin = margin.to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
lambda1 = lambda step: (step / warmup) if step < warmup else 0.5 * (math.cos((step - warmup) / (epochs * len_train - warmup) * math.pi) + 1)
optimizer = optim.SGD([
        {'params': backbone.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=learning_rate, momentum=0.9, nesterov=True)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

if mixed_precision:
    print("mixed_precision")
    [backbone, margin], optimizer = amp.initialize([backbone, margin], optimizer, opt_level='O1', verbosity=0)

iter_idx = 0
for epoch in range(epochs):
    print('\nEpoch: {}'.format(epoch + 1))
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    since = time.time()
    for inputs, targets in train_loader:
        backbone.train()
        margin.train()
        inputs, targets = inputs.to(device), targets.to(device)
        feature = backbone(inputs)
        outputs = margin(feature, targets, device=device, mixed_precision=mixed_precision)
        loss = criterion(outputs, targets)
        if mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if iter_idx % accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        batch_idx += 1
        iter_idx += 1
        if batch_idx % 20 == 0:
            time_total = time.time() - since
            since = time.time()
            train_status = f'Epoch[{epoch + 1}] Batch [{batch_idx - 20}-{batch_idx}] ' \
                f'Learning Rate: {optimizer.param_groups[0]["lr"]:f} ' \
                f'Speed: {batch_size * 20 / time_total:.2f} samples/sec Acc={correct / total:f} ' \
                f'Loss={train_loss / (batch_idx + 1):f}'
            writer.add_scalar('data/learning', optimizer.param_groups[0]["lr"], global_step=iter_idx)
            writer.add_scalar('data/accuracy', correct / total, global_step=iter_idx)
            writer.add_scalar('data/loss', train_loss / (batch_idx + 1), global_step=iter_idx)
            print(train_status)

        if iter_idx % 2000 == 0:
            writer.add_embedding(feature, metadata=targets, label_img=inputs, global_step=iter_idx)
            backbone.eval()
            embedding_list = [np.zeros((len_lfw, feature_dim)), np.zeros((len_lfw, feature_dim))]
            with torch.no_grad():
                test_batch = 0
                for image1, image2 in val_loader:
                    image1, image2 = image1.to(device), image2.to(device)
                    embedding1, embedding2 = backbone(image1), backbone(image2)
                    embedding1, embedding2 = embedding1.detach().cpu().numpy(), embedding2.detach().cpu().numpy()
                    embedding_list[0][test_batch * batch_size:(test_batch + 1) * batch_size] = embedding1
                    embedding_list[1][test_batch * batch_size:(test_batch + 1) * batch_size] = embedding2
                    test_batch += 1
            embeddings = embedding_list[0] + embedding_list[1]
            embeddings = sklearn.preprocessing.normalize(embeddings)
            _, _, accuracy, val, val_std, far = evaluate(embeddings, val_dataset.issame_list)
            acc2, std2 = np.mean(accuracy), np.std(accuracy)
            print(acc2)

            if acc2 > best_acc:
                backbone_state = {
                    'net': backbone.state_dict(),
                    'acc': acc2,
                    'epoch': epoch,
                }
                margin_state = {
                    'net': margin.state_dict(),
                    'acc': acc2,
                    'epoch': epoch,
                }
                if not os.path.isdir('weights'):
                    os.mkdir('weights')
                torch.save(backbone_state, './weights/backbone.pth')
                torch.save(margin_state, './weights/margin.pth')
                best_acc = acc2
                print('Saving..\n Better Accuracy: {:.4f}%'.format(best_acc*100))
            else:
                print('Better Accuracy: {:.4f}%'.format(best_acc * 100))
        scheduler.step()

print('Best Accuracy: {:.4f}%'.format(best_acc*100))
