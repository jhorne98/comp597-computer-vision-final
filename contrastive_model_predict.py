import argparse
import re

from cub2011 import Cub2011

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

device = torch.device('cpu')
model_types = [(models.resnet18, "resnet18"), (models.resnet50, "resnet50"), (models.resnet101, "resnet101")]

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

        print(model.encoder.parameters())

        with torch.no_grad():
            out = model(inp)
            probs = torch.softmax(out, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs.max().item()

        print(f"predicted class: {pred_class}, confidence: {confidence:.4f}")