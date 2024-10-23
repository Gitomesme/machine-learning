import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

train_dataset = datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root="./data",train=False,download=True,transform=transform)

train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)

class SimpleNN(nn.Module):
	def __init__(self):
		super(SimpleNN, self).__init__()
		self.fc1 = nn.Linear(28 * 28, 128)  # couche d'entrée (image 28x28 pixels)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)  # sortie pour 10 classes (les chiffres 0 à 9)

	def forward(self, x):
		x = x.view(-1, 28 * 28)  # Aplatir l'image 28x28 en un vecteur
		x = functional.relu(self.fc1(x))
		x = functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# Initialiser le modèle
model = SimpleNN()

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
def train_model(model, train_loader, criterion, optimizer, epochs=2):
    for epoch in range(epochs):
        model.train()  # Mode entraînement
        running_loss = 0.0

        for images, labels in train_loader:
            # Réinitialiser les gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

# Appel de la fonction d'entraînement
train_model(model, train_loader, criterion, optimizer)

def evaluate_model(model, test_loader):
    model.eval()  # Mode évaluation
    correct = 0
    total = 0

    with torch.no_grad():  # Pas besoin de calculer les gradients en mode évaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

# Appel de la fonction d'évaluation
evaluate_model(model, test_loader)

torch.save(model.state_dict(), "models/cnn_model.pth")
print("Modèle sauvegardé dans models/cnn_model.pth")

# Fonction pour afficher les images
def imshow(img):
    img = img / 2 + 0.5  # Dénormaliser
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Afficher des exemples du DataLoader
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Affiche les premières images
imshow(torchvision.utils.make_grid(images))

# Affiche les labels correspondants
print('Labels:', ' '.join(f'{labels[j].item()}' for j in range(8)))