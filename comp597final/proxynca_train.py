import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models
import sklearn.preprocessing
from tqdm import tqdm

from cub2011 import Cub2011

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frozen_proxynca_model_resnet50.pth')
LOG_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frozen_training_log.txt')

#transform images for proxy
proxy_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes=range(nb_classes))
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    return torch.FloatTensor(T).to(device)


class ProxyNCA(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, smoothing_const=0.1,
                 scaling_x=3.0, scaling_p=3.0):
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


if __name__ == '__main__':
    #training loop for proxynca
    full_train_dataset = Cub2011(root='./cub2011', train=True, download=True, transform=proxy_transform)
    #Only keep samples with target in 0-99 (first 100 classes)
    train_indices = [i for i, (_, target) in enumerate(full_train_dataset) if 0 <= target < 100]
    proxy_train_dataset = Subset(full_train_dataset, train_indices)
    proxy_train_loader = DataLoader(proxy_train_dataset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

    #just use resnet50 for now
    for model_type in [(models.resnet50, "resnet50")]:
        base = model_type[0](weights=models.ResNet50_Weights.DEFAULT)
        embedding_dim = 64
        base.fc = nn.Linear(base.fc.in_features, embedding_dim)
        #freeze all layers except the final (new) fc layer
        for name, param in base.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        encoder = base.to(device)

        proxy_loss_fn = ProxyNCA(
            num_classes=100,  #100 classes for training split
            embedding_dim=embedding_dim,
        ).to(device)

        optimizer = torch.optim.Adam([
            {'params': encoder.parameters(), 'lr': 1e-4},
            {'params': proxy_loss_fn.parameters(), 'lr': 1.0}
        ], weight_decay=1e-4)

        with open(LOG_PATH, 'w') as log:
            log.write(f"Training ProxyNCA | model: {model_type[1]}\n")
            log.write("epoch, loss\n")

        for epoch in range(50):
            encoder.train()
            proxy_loss_fn.train()
            total_loss = 0
            for images, labels in tqdm(proxy_train_loader):
                images, labels = images.to(device), labels.to(device)
                z = encoder(images)

                loss = proxy_loss_fn(z, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(proxy_train_loader)
            print(f"epoch {epoch+1} | Loss: {avg_loss:.4f}")

            #write epoch and loss to log file
            with open(LOG_PATH, 'a') as log:
                log.write(f"{epoch+1}, {avg_loss:.4f}\n")

        torch.save({'encoder': encoder.state_dict(), 'proxies': proxy_loss_fn.state_dict()}, SAVE_PATH)
        print(f"Model saved to: {SAVE_PATH}")
        print(f"Training log saved to: {LOG_PATH}")
