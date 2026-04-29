import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from lightly.models.modules import heads
from lightly.loss import NTXentLoss
from lightly.transforms import SimCLRTransform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageNet
from cub2011 import Cub2011 # Adapted from https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/cub2011.py
import torchvision.models as models
import random
from tqdm import tqdm
import sklearn.preprocessing
import logging
import sklearn.cluster
import sklearn.metrics.cluster
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = [10, 50]
model_types = [(models.resnet18, "resnet18"), (models.resnet50, "resnet50"), (models.resnet101, "resnet101")]

simclr_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(kernel_size=3),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

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

def extract_embeddings_labels(model, loader):
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model(x)
            embeddings.append(z.cpu())
            labels.append(y)
    return torch.cat(embeddings), torch.cat(labels)

def nt_xent_loss(z_i, z_j, temperature=0.05):
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T)
    N = z_i.shape[0]

    mask = (~torch.eye(2*N, dtype=bool)).to(z.device)
    sim = similarity / temperature
    exp_sim = torch.exp(sim) * mask

    positive_sim = torch.exp(F.cosine_similarity(z_i, z_j) / temperature)
    positives = torch.cat([positive_sim, positive_sim], dim=0)

    denominator = exp_sim.sum(dim=1)
    loss = -torch.log(positives / denominator)
    return loss.mean()

train_dataset = Cub2011(root=str('./cub2011'), train=True, download=True)
contrastive_dataset = SimCLRDataset(train_dataset, simclr_transform)
train_loader = DataLoader(contrastive_dataset, batch_size=128, shuffle=True, num_workers=2)

for num_epochs in epochs:
    for model_type in model_types:
        model = SimCLRModel(model_type[0], weights=None).to(device)
        #https://discuss.pytorch.org/t/passing-to-the-optimizers-frozen-parameters/83358
        #https://discuss.pytorch.org/t/how-can-i-exclude-some-parameters-in-optimizer-during-training/90208
        #https://discuss.pytorch.org/t/best-practice-for-freezing-layers/58156
        # Freeze the encoder weights, only train the projection head
        #for param in model.encoder.parameters():
        #        param.requires_grad = False
        
        optimizer = torch.optim.Adam(model.parameters(), lr=.005)
        #https://www.kozodoi.me/blog/gradient-accumulation-in-pytorch
        #https://wandb.ai/wandb_fc/tips/reports/How-To-Implement-Gradient-Accumulation-in-PyTorch--VmlldzoyMjMwOTk5
        accumulate_batches = 4

        print("Epochs: " + str(num_epochs) + ", Model Type: " + model_type[1])
        
        losses = []
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            for i, samples in enumerate(tqdm(train_loader)):
                x_i, x_j = samples
                x_i = x_i.to(device)
                x_j = x_j.to(device)
                z_i = model(x_i)
                z_j = model(x_j)
        
                loss = nt_xent_loss(z_i, z_j)
                loss = loss/accumulate_batches
                loss.backward()
        
                if (i+1) % accumulate_batches == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        
                total_loss += loss.item() * accumulate_batches
        
        
            if(len(train_loader) % accumulate_batches != 0):
                optimizer.step()
                optimizer.zero_grad()
        
            losses.append(total_loss/len(train_loader))
            print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

            if epoch % 5 == 0:
                torch.save(model.state_dict(), "SimCLR_unfrozen" + model_type[1] + "_" + str(num_epochs) + "e_512b_.005lr.pth")

        with open("simclr_losses_unfrozen.txt", "a", encoding='utf-8') as f:
            f.write("SimCLR_" + model_type[1] + "_" + str(num_epochs) + "e_512b_.005lr.pth: " + str(losses) + "\n")

        del model
        del optimizer
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

print("Training completed!")