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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda')
model_types = [(models.resnet18, "resnet18"), (models.resnet50, "resnet50"), (models.resnet101, "resnet101")]

labels_path = "cub2011/CUB_200_2011/classes.txt"

simclr_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(kernel_size=3),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

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
    
class LinearClassifier(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.encoder = model.encoder

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            feat = self.encoder(dummy).flatten(1)
            num_feat = feat.shape[1]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, im):
        with torch.no_grad():
            feats = self.encoder(im).flatten(1)
        logits = self.classifier(feats)
        return logits

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
    
class SimCLRLabeledDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label

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
    parser.add_argument("-t", "--train", type=bool, default=False)
    parser.add_argument("-m", "--base_model", required=True)
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
    model_file = str(args.base_model)
    input = args.input
    base_model = re.search(r"resnet\d+", model_file)
    base_model = base_model.group(0) if base_model else None
    model_type = next((m for m in model_types if m[1] == base_model), None)

    if args.train:
        if "SimCLR" in model_file:
            train_dataset = Cub2011(root=str('./cub2011'), train=True, download=True)
            label_dataset = SimCLRLabeledDataset(train_dataset, simclr_transform)
            train_loader = DataLoader(label_dataset, batch_size=256, shuffle=True, num_workers=2)

            val_dataset = Cub2011(root=str('./cub2011'), train=False, download=False)
            val_label_dataset = SimCLRLabeledDataset(val_dataset, simclr_transform)
            val_loader = DataLoader(val_label_dataset, batch_size=256, shuffle=True, num_workers=2)

            # this isn't strictly necessary, but I think it will be consistent with actually using the model for ProxyNCA
            simclr_model = SimCLRModel(model_type[0], projection_dim=128).to(device)
            simclr_model.load_state_dict(torch.load(model_file, map_location=device))
            simclr_model.eval()

            linear_classifier = LinearClassifier(simclr_model, 200).to(device)

            # freeze encoder
            for param in linear_classifier.encoder.parameters():
                param.requires_grad = False
        
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(linear_classifier.classifier.parameters(), lr=.03, momentum=0.9)

            epochs = 20
            for epoch in range(epochs):
                linear_classifier.train()
                total_loss = 0.0
                correct = 0
                total = 0
                for i, samples in enumerate(tqdm(train_loader)):
                    imgs, targets = samples
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    logits = linear_classifier(imgs)
                    #print(len(targets))
                    loss = criterion(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * imgs.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += imgs.size(0)

                train_accuracy = correct / total
                avg_loss = total_loss / total

                linear_classifier.eval()
                vcorrect = 0
                vtotal = 0
                with torch.no_grad():
                    for imgs, targets in val_loader:
                        imgs = imgs.to(device)
                        targets = targets.to(device)
                        logits = linear_classifier(imgs)
                        preds = logits.argmax(dim=1)
                        vcorrect += (preds == targets).sum().item()
                        vtotal += imgs.size(0)
                val_acc = vcorrect / vtotal
                print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  train_acc={train_accuracy:.4f}  val_acc={val_acc:.4f}")

                if epoch % 5 == 0:
                    torch.save(linear_classifier.state_dict(), "SimCLR_linear_classifier.pth")

    else:
        img = Image.open(input).convert("RGB")
        inp = evaluation_transform(img).unsqueeze(0)

        if "SimCLR" in model_file:
            model = SimCLRModel(model_type[0], projection_dim=128).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.fc = nn.Linear(model.num_ftrs, 200)
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
