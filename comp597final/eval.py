import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models
import sklearn.preprocessing
import sklearn.cluster
import sklearn.metrics.cluster

from cub2011 import Cub2011

#CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proxynca_model_resnet50.pth')
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frozen_proxynca_model_resnet50.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

evaluation_transform_NCA = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes=range(nb_classes))
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    return torch.FloatTensor(T).to(device)


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))

def assign_by_euclidian_at_k(X, T, k):
    """
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    distances = torch.cdist(X, X)
    # get nearest points
    indices = distances.topk(k + 1, largest=False)[1][:, 1: k + 1]
    return np.array([[T[i] for i in ii] for ii in indices])

def cluster_by_kmeans(X, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    return sklearn.cluster.KMeans(nb_clusters, n_init='auto').fit(X).labels_

def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys)

def predict_batchwise(model, dataloader): #returns embeddings and targets for all samples in dataloader, without shuffling
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        for batch in dataloader:
            for i, J in enumerate(batch):
                if i == 0:
                    J = J.to(list(model.parameters())[0].device)
                    J = model(J).cpu()
                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training)
    return [torch.stack(A[i]) for i in range(len(A))]

def evaluate(model, dataloader, nb_classes, with_nmi=True): #returns recall at k for k in [1, 2, 4, 8, 10], and optionally NMI
    X, T, *_ = predict_batchwise(model, dataloader) #x is image embeddings, t is target labels, *_ is for ignoring any other outputs from predict_batchwise
    if with_nmi:
        nmi = calc_normalized_mutual_information(
            T,
            cluster_by_kmeans(X, nb_classes)
        )
        print("NMI: {:.3f}".format(nmi * 100))
    Y = assign_by_euclidian_at_k(X, T, 8)
    Y = torch.from_numpy(Y)
    recall = []
    for k in [1, 2, 4, 8, 10]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    if with_nmi:
        return recall, nmi
    else:
        return recall


if __name__ == '__main__':
    full_test_dataset = Cub2011(root='./cub2011', train=False, download=True, transform=evaluation_transform_NCA)
    # Only keep samples with target in 100-199 (last 100 classes)
    test_indices = [i for i, (_, target) in enumerate(full_test_dataset) if 100 <= target < 200]
    test_dataset = Subset(full_test_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


    base_model = models.resnet50(weights=None)
    embedding_dim = 64
    base_model.fc = nn.Linear(base_model.fc.in_features, embedding_dim)
    encoder_model = base_model.to(device)


    # Load the saved state dictionaries for encoder
    print(f"Loading checkpoint: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    encoder_model.load_state_dict(checkpoint['encoder'])
    encoder_model.eval()

    print("Evaluating...")
    evaluate(encoder_model, test_loader, nb_classes=100)




