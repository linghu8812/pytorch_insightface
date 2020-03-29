import os
import time
import numpy as np
import torch
import sklearn
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy import interpolate
from val_dataloader import FaceRecognitionValDataset, val_dataset_transform, DataLoader
from model.fresnetv2 import resnet18


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        # print('threshold', thresholds[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def get_val_features(backbone, val_loader, issame_list, dataset_name, len_val, feature_dim, batch_size, device):
    embedding_list = [np.zeros((len_val, feature_dim)), np.zeros((len_val, feature_dim))]
    print(f'Evaluating {dataset_name} ...')
    start_time = time.time()
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
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    print(f'Evaluate cost {time.time() - start_time:.2f} sec')
    print(f'Accuracy is {acc2*100:.4f}, Standard is {std2:f}')
    return acc2, std2


def main():
    batch_size = 128
    feature_dim = 512
    lfw_dataset = FaceRecognitionValDataset(bin_file='./dataset/lfw.bin', transform=val_dataset_transform)
    len_lfw = len(lfw_dataset)
    lfw_loader = DataLoader(lfw_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)

    cfp_fp_dataset = FaceRecognitionValDataset(bin_file='./dataset/cfp_fp.bin', transform=val_dataset_transform)
    len_cfp_fp = len(cfp_fp_dataset)
    cfp_fp_loader = DataLoader(cfp_fp_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)

    agedb_30_dataset = FaceRecognitionValDataset(bin_file='./dataset/agedb_30.bin', transform=val_dataset_transform)
    len_agedb_30 = len(agedb_30_dataset)
    agedb_30_loader = DataLoader(agedb_30_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('==> Building model ...')
    backbone = resnet18(num_classes=feature_dim)
    weights_file = 'weights/backbone.pth'
    if os.path.exists(weights_file):
        backbone.load_state_dict(torch.load(weights_file)['net'])
    backbone = backbone.to(device)
    backbone.eval()
    get_val_features(backbone, lfw_loader, lfw_dataset.issame_list, lfw_dataset.dataset_name, len_lfw, feature_dim,
                     batch_size, device)
    get_val_features(backbone, cfp_fp_loader, cfp_fp_dataset.issame_list, cfp_fp_dataset.dataset_name, len_cfp_fp,
                     feature_dim, batch_size, device)
    get_val_features(backbone, agedb_30_loader, agedb_30_dataset.issame_list, agedb_30_dataset.dataset_name,
                     len_agedb_30, feature_dim, batch_size, device)
    return


if __name__ == '__main__':
    main()
