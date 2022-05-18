import pickle
import os
import sys

import pandas as pd
from sympy import N
from torchvision.io import read_image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as tud
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ssl
from PIL import Image
import torchvision.transforms.functional as TF
import gradio as gr
import os
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.reshape(-1, 16 * 5 * 5)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class MyApp:
    def __init__(self):
        super().__init__()
        self.epoch = 25
        self.batch_size = 64
        self.lr = 0.001
        self.model = MyModel()

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.labels = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                                     download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                                    download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)


app = MyApp()


def save_model(my_model, path):
    torch.save(my_model.state_dict(), path)


def plot_accuracy(train_accuracies):
    plt.plot(train_accuracies)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def plot_loss(train_loss):
    plt.plot(train_loss)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def train_model(device):
    loss_fonct = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(app.model.parameters(), lr=0.01)

    train_accuracies = np.zeros(app.epoch)
    train_loss = []
    count = 0

    for epoch in tqdm(range(app.epoch)):
        total_train, correct_train = 0, 0
        for batch in tqdm(app.trainloader):
            images, labels = batch
            images = images.to(device=device)
            labels = labels.to(device=device)

            output = app.model.forward(images)
            loss = loss_fonct(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if count % 10 == 0:
                train_loss.append(loss)
            count = count + 1

            _, predicted = torch.max(output.data, 1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            train_accuracies[epoch] = correct_train / total_train * 100
        print("Accuracy:", correct_train / total_train * 100, "%\n\n")

    save_model(app.model, "cifar10-model.pth")
    plot_accuracy(train_accuracies)


def test_image(path):
    image = Image.open(path)
    transform = transforms.ToTensor()
    x = transform(image)
    x = x.unsqueeze(0)
    # x.reshape(1, 1, 32, 32)
    out = app.model(x)
    _, pred = torch.max(out, dim=1)
    return app.classes[pred]


def check_weights():
    ssl._create_default_https_context = ssl._create_unverified_context

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("will run on GPU")
    else:
        print("will run on CPU")

    app.model.to(device)

    print("Loading model...")
    if os.path.isfile("cifar10-model.pth"):
        print("Model found")
        app.model.load_state_dict(torch.load("cifar10-model.pth"))
        print("Model loaded")
    else:
        print("No saved model found\nLet's go train !")
        train_model(device)
    return app


def predict(image):
    out = app.model(image.reshape(1, 3, 32, 32))
    _, pred = torch.max(out, dim=1)
    return app.classes[pred]


IMG_SIZE = 32 if torch.cuda.is_available() else 32
COMPOSED_TRANSFORMERS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

NORMALIZE_TENSOR = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def np_array_to_tensor_image(img, width=IMG_SIZE, height=IMG_SIZE, device='cpu'):
    image = Image.fromarray(img).convert('RGB').resize((width, height))
    image = COMPOSED_TRANSFORMERS(image).unsqueeze(0)

    return image.to(device, torch.float)


def normalize_tensor(tensor: torch.tensor) -> torch.tensor:
    return NORMALIZE_TENSOR(tensor)


def sketch_recognition(img):
    img = np_array_to_tensor_image(img)
    img = normalize_tensor(img)
    result = predict(img)
    print(result)
    return result
    pass


def segment(img):
    img = np_array_to_tensor_image(img)
    img = normalize_tensor(img)
    result = predict(img)
    print(result)
    return result
    pass


def check_sample(image):
    print("Predicted: " + predict(app.testset[image][0]))
    print("Desired output: " + app.classes[app.testset[image][1]])


def my_app(input_type):
    check_weights()
    if input_type == "1":
        gr.Interface(fn=sketch_recognition, inputs=["sketchpad"], outputs="label").launch(share=True)
    elif input_type == "2":
        gr.Interface(fn=segment, inputs=["image"], outputs="label").launch(share=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        my_app(sys.argv[1])
    else:
        print("Please specify the input type")