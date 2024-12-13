import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import norse.torch as snn
import matplotlib.pyplot as plt

# load le dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # mettre entre -1 et 1
])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("loaded")

# encoder le dataset en spike
# plus le pixel est clair plus il y a de chance qu'il y ait un spike
def poisson_encoder(image, time: int):
    flattened = image.view(-1)  #  28x28 a 784 x 1
    # en gros spike est un tenseur d'une taille de time x 784
    # rand cree un chiffre aleatoire entre 0 et 1
    # on compare ce chiffre a flattened.unsqueeze(0) qui est un tenseur de 1 x 784 qui représente les pixels
    # si le chiffre aleatoire est plus petit que le pixel alors on met un spike
    spikes = torch.rand(time, *flattened.shape) < flattened.unsqueeze(0)
    return spikes.float()


example_image, example_label = next(iter(train_loader))
spike_train = poisson_encoder(example_image[0], time=100)

print("Spike train shape:", spike_train.shape)


# Visualisation du spike train
plt.figure(figsize=(10, 5))
for neuron_id, spikes in enumerate(spike_train.T):
    spike_times = (spikes > 0).nonzero(as_tuple=True)[0]
    plt.scatter(spike_times, [neuron_id] * len(spike_times), s=1)
plt.xlabel("temps")
plt.ylabel("neurone")
plt.show()
