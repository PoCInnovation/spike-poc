import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import norse.torch as snn
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: load le dataset
# -----------------------------

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # on normalise toutes les valeurs
    ]
)

train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("MNIST dataset loaded successfully.")


# -----------------------------
# Step 2: on encode en spike train
# -----------------------------
def poisson_encoder(image, time: int):
    flattened = image.view(-1)  # 28x28 -> 784
    spikes = torch.rand(time, *flattened.shape) < flattened.unsqueeze(0)
    return spikes.float()


# Test et visu de l'encodage en spike train
# example_image, example_label = next(iter(train_loader))
# spike_train = poisson_encoder(example_image[0], time=100)

# plt.figure(figsize=(10, 5))
# for neuron_id, spikes in enumerate(spike_train.T):
#     spike_times = (spikes > 0).nonzero(as_tuple=True)[0]
#     plt.scatter(spike_times, [neuron_id] * len(spike_times), s=1)
# plt.xlabel("Time (ms)")
# plt.ylabel("Neurone")
# plt.show()

# print("Spike train shape:", spike_train.shape)


# -----------------------------
# Step 3: creation du SNN
# -----------------------------
class SpikingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Linear(784, 128)
        self.lif_hidden = snn.LIFCell()

        self.hidden_to_output = nn.Linear(128, 10)
        self.lif_output = snn.LIFCell()

    def forward(self, x, state_hidden=None, state_output=None):
        x = self.input_to_hidden(x)
        spikes_hidden, state_hidden = self.lif_hidden(x, state_hidden)

        x = self.hidden_to_output(spikes_hidden)
        spikes_output, state_output = self.lif_output(x, state_output)

        return spikes_output, state_hidden, state_output

model = SpikingNN()
print(model)

# -----------------------------
# Test du SNN
# -----------------------------

# batch_size = 64
# time_steps = 100
# spike_input = torch.rand(time_steps, batch_size, 784)  # Random spike trains

# # Properly initialize states
# state_hidden = model.lif_hidden.initial_state(torch.zeros([batch_size, 128]))
# state_output = model.lif_output.initial_state(torch.zeros([batch_size, 10]))

# # Forward pass over time
# for t in range(time_steps):
#     spike_output, state_hidden, state_output = model(
#         spike_input[t], state_hidden, state_output
#     )

# print("Output shape:", spike_output.shape)

