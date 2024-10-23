import matplotlib.pyplot as plt
import numpy as np

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
