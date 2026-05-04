from linear_classifier import ProxyNCA, SimCLRModel, SimCLRLabeledDataset, LinearClassifier, evaluation_transform
import torch
import torch.nn as nn
import torchvision.models as models
import random
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels_path = "cub2011/CUB_200_2011/classes.txt"
images_path = "cub2011/CUB_200_2011/images.txt"

rounds = 2000

def main():
    labels = []
    with open(labels_path, "r", encoding='utf-8') as f:
        for line in f:
            labels.append(line.rstrip("\n"))

    filenames = []
    with open(images_path, "r", encoding='utf-8') as f:
        for line in f:
            _, _, filename = line.partition(" ")
            filenames.append("cub2011/CUB_200_2011/images/" + filename.rstrip("\n"))

    subset_images = random.sample(filenames, rounds)

    simclr_encoder = SimCLRModel(models.resnet50, projection_dim=128).to(device)
    simclr_encoder.load_state_dict(torch.load("SimCLR_resnet50_50e_512b_.005lr.pth", map_location=device))
    simclr_encoder = simclr_encoder.encoder

    proxynca_encoder = models.resnet50(weights=None)
    proxynca_encoder.fc = nn.Linear(proxynca_encoder.fc.in_features, 64)
    proxynca_encoder.to(device)
    proxynca_encoder.load_state_dict(torch.load("ProxyNCA_model_resnet50.pth", map_location=device)['encoder'])

    simclr_encoder.eval()
    proxynca_encoder.eval()

    simclr_classifier = LinearClassifier(simclr_encoder, 200).to(device)
    simclr_classifier.load_state_dict(torch.load("SimCLR_Classifier_resnet50_50e.pth", map_location=device))

    proxynca_classifier = LinearClassifier(proxynca_encoder, 200).to(device)
    proxynca_classifier.load_state_dict(torch.load("ProxyNCA_Classifier_resnet50_50e.pth", map_location=device))

    total_simclr_correct = 0
    total_proxynca_correct = 0

    total_simclr_confidence = 0.0
    total_proxynca_confidence = 0.0

    for i, img_filename in enumerate(subset_images):
        print(f"Round {i} actual: {img_filename}")
        img = Image.open(img_filename).convert("RGB")
        inp = evaluation_transform(img).unsqueeze(0).to(device)

        actual_class = int(img_filename[img_filename.find("images/")+len("images/"):img_filename.find(".")])        

        with torch.no_grad():
            simclr_pred = simclr_classifier(inp)
            proxynca_pred = proxynca_classifier(inp)

            simclr_probs = torch.softmax(simclr_pred, dim=1)
            simclr_pred_class = simclr_probs.argmax(dim=1).item()
            simclr_confidence = simclr_probs.max().item()

            proxynca_probs = torch.softmax(proxynca_pred, dim=1)
            proxynca_pred_class = proxynca_probs.argmax(dim=1).item()
            proxynca_confidence = proxynca_probs.max().item()

        print(f"Round {i}: SimCLR predicted class: {labels[simclr_pred_class]}, SimCLR confidence: {simclr_confidence:.4f}")
        print(f"Round {i}: ProxyNCA predicted class: {labels[proxynca_pred_class]}, ProxyNCA confidence: {proxynca_confidence:.4f}")

        total_simclr_correct += int(simclr_pred_class+1 == actual_class)
        total_proxynca_correct += int(proxynca_pred_class+1 == actual_class)

        total_simclr_confidence += simclr_confidence
        total_proxynca_confidence += proxynca_confidence

    print("\n")
    print(f"Total SimCLR correct: {total_simclr_correct} ({(total_simclr_correct/rounds)*100:.1f})%")
    print(f"Total ProxyNCA correct: {total_proxynca_correct} ({(total_proxynca_correct/rounds)*100:.1f})%")

    print(f"Average SimCLR confidence: {(total_simclr_confidence/rounds)*100:.4f}%")
    print(f"Average ProxyNCA confidence: {(total_proxynca_confidence/rounds)*100:.4f}%")

if __name__ == '__main__':
    main()