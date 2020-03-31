import os
import time
import math
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model.fresnetv2 import resnet18
from margin.ArcLoss import ArcLossMargin
from margin.MixupArcLoss import MixupArcLossMargin
from margin.LabelSmoothing import LabelSmoothing
from train_dataloader import FaceRecognitionTrainDataset, train_dataset_transform, DataLoader
from val_dataloader import FaceRecognitionValDataset, val_dataset_transform
from verification import get_val_features
from mixup import mixup_data, mixup_criterion

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    mixed_precision = False
mixup = 1
alpha = 1.5
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
learning_rate = 0.01
batch_size = 256
accumulate = 2
epochs = 50
feature_dim = 512
warmup = 4000
writer = SummaryWriter(log_dir='logs2')

# Data
print('==> Preparing data ...')

train_dataset = FaceRecognitionTrainDataset(train_list='./dataset/train2.lst', transform=train_dataset_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8)
len_train = len(train_loader)

lfw_dataset = FaceRecognitionValDataset(bin_file='./dataset/lfw.bin', transform=val_dataset_transform)
len_lfw = len(lfw_dataset)
lfw_loader = DataLoader(lfw_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)

cfp_fp_dataset = FaceRecognitionValDataset(bin_file='./dataset/cfp_fp.bin', transform=val_dataset_transform)
len_cfp_fp = len(cfp_fp_dataset)
cfp_fp_loader = DataLoader(cfp_fp_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)

agedb_30_dataset = FaceRecognitionValDataset(bin_file='./dataset/agedb_30.bin', transform=val_dataset_transform)
len_agedb_30 = len(agedb_30_dataset)
agedb_30_loader = DataLoader(agedb_30_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)

print('==> Building model ...')
backbone = resnet18(num_classes=feature_dim)
backbone = backbone.to(device)
if mixup:
    margin = MixupArcLossMargin(input_dim=feature_dim, class_number=train_dataset.class_nums)
    margin = margin.to(device)
else:
    margin = ArcLossMargin(input_dim=feature_dim, class_number=train_dataset.class_nums)
    margin = margin.to(device)

criterion = LabelSmoothing().to(device)
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
        if mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha)
            feature = backbone(inputs)
            outputs = margin(feature, targets_a, targets_b, lam, device=device, mixed_precision=mixed_precision)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
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
        train_loss += loss.item() / batch_size
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if mixup:
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        else:
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
            writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], global_step=iter_idx)
            writer.add_scalar('train_accuracy', correct / total, global_step=iter_idx)
            writer.add_scalar('train_loss', train_loss / (batch_idx + 1), global_step=iter_idx)
            print(train_status)

        if iter_idx % 2000 == 0:
            writer.add_embedding(feature, metadata=targets, label_img=inputs, global_step=iter_idx)
            backbone.eval()
            lfw_acc, _ = get_val_features(backbone, lfw_loader, lfw_dataset.issame_list, lfw_dataset.dataset_name,
                                          len_lfw, feature_dim, batch_size, device)
            cfp_fp_acc, _ = get_val_features(backbone, cfp_fp_loader, cfp_fp_dataset.issame_list,
                                             cfp_fp_dataset.dataset_name, len_cfp_fp, feature_dim, batch_size, device)
            agedb_30_acc, _ = get_val_features(backbone, agedb_30_loader, agedb_30_dataset.issame_list,
                                               agedb_30_dataset.dataset_name, len_agedb_30, feature_dim, batch_size,
                                               device)

            writer.add_scalars('test_accuracy', {'lfw_accuracy': lfw_acc, 'cfp_fp_accuracy': cfp_fp_acc,
                                                 'agedb_30_accuracy': agedb_30_acc}, global_step=iter_idx)

            if agedb_30_acc > best_acc:
                backbone_state = {
                    'net': backbone.state_dict(),
                    'acc': agedb_30_acc,
                    'epoch': epoch,
                }
                margin_state = {
                    'net': margin.state_dict(),
                    'acc': agedb_30_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('weights'):
                    os.mkdir('weights')
                torch.save(backbone_state, './weights/backbone.pth')
                torch.save(margin_state, './weights/margin.pth')
                best_acc = agedb_30_acc
                print('Saving..\n Better Accuracy: {:.4f}%'.format(best_acc*100))
            else:
                print('Better Accuracy: {:.4f}%'.format(best_acc * 100))
        scheduler.step()

print('Best Accuracy: {:.4f}%'.format(best_acc*100))
