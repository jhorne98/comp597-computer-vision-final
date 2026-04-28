import argparse
import re

from cub2011 import Cub2011

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import sklearn.preprocessing
from PIL import Image

device = torch.device('cpu')
model_types = [(models.resnet18, "resnet18"), (models.resnet50, "resnet50"), (models.resnet101, "resnet101")]

labels_path = "cub2011/CUB_200_2011/classes.txt"

evaluation_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

class SimCLRModel(nn.Module):
    def __init__(self, model, weights='DEFAULT', projection_dim=128):
        super().__init__()
        base_model = model(weights=weights)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Identity()
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z

class SimCLRDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        xi = self.transform(image)
        xj = self.transform(image)
        return xi, xj

    def __len__(self):
        return len(self.dataset)

def extract_embeddings_labels(model, loader):
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model(x)
            embeddings.append(z.cpu())
            labels.append(y)
    return torch.cat(embeddings), torch.cat(labels)

def recall_at_k(embeddings, labels, k=3):
    correct = 0
    numEmbeddings = len(labels)
    for i in range(numEmbeddings):
        current_embedding = embeddings[i]
        dist = torch.norm(embeddings - current_embedding, dim=1, p = None)
        dist[i] = float('inf')
        k_nearest = dist.topk(k, largest=False)
        for idx in k_nearest.indices:
            if labels[idx] == labels[i]:
                correct += 1
                break
    return correct / numEmbeddings

def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes=range(nb_classes))
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    return torch.FloatTensor(T).to(device)

class ProxyNCA(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, smoothing_const=0.1,
                 scaling_x=1.0, scaling_p=3.0):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim) / 8)
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p
        self.num_classes = num_classes

    def forward(self, z, labels):
        P = F.normalize(self.proxies, p=2, dim=-1) * self.scaling_p
        Z = F.normalize(z, p=2, dim=-1) * self.scaling_x

        D = torch.cdist(Z, P) ** 2

        T = binarize_and_smooth_labels(labels, self.num_classes, self.smoothing_const)
        loss = torch.sum(-T * F.log_softmax(-D, dim=-1), dim=-1)
        return loss.mean()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="constrastive_model_predict"
    )
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-i", "--input", required=True)
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    labels = []
    with open(labels_path, "r", encoding='utf-8') as f:
        for line in f:
            labels.append(line.rstrip("\n"))

    args = main()
    model_file = str(args.model)
    input = args.input
    base_model = re.search(r"resnet\d+", model_file)
    base_model = base_model.group(0) if base_model else None
    model_type = next((m for m in model_types if m[1] == base_model), None)

    img = Image.open(input).convert("RGB")
    inp = evaluation_transform(img).unsqueeze(0)

    if "SimCLR" in model_file:
        model = SimCLRModel(model_type[0], projection_dim=128).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.fc = nn.Linear(128, 200)
        model.eval

        #print(model.encoder.parameters())

        with torch.no_grad():
            out = model(inp)
            probs = torch.softmax(out, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs.max().item()

        print(f"predicted class: {labels[pred_class]}, confidence: {confidence:.4f}")
    elif "ProxyNCA" in model_file:
        base_model = model_type[0](weights='DEFAULT')
        embedding_dim = base_model.fc.in_features
        base_model.fc = nn.Linear(128, 200)
        encoder_model = base_model.to(device)

        proxy_nca_loss_fn = ProxyNCA(num_classes=100, embedding_dim=embedding_dim).to(device)
        
        checkpoint = torch.load(model_file, map_location=device)
        encoder_model.load_state_dict(checkpoint['encoder'])
        proxy_nca_loss_fn.load_state_dict(checkpoint['proxies'])
