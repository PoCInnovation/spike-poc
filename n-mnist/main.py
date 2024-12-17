from operator import index
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import norse.torch as snn

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

train_loader = DataLoader(train_dataset, batch_size=6000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6000, shuffle=False)


# -----------------------------
# Step 2: on encode en spike train
# -----------------------------
def poisson_encoder(images, time_steps: int):
    batch_size = images.shape[0]
    images = images.view(batch_size, -1) # 28x28 -> 784
    spikes = torch.rand(time_steps, batch_size, images.shape[1]).to(images.device)
    spikes = (spikes < images.unsqueeze(0)).float()
    return spikes

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


# initialisation
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")
model = SpikingNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

time_steps = 100 # duree des spike trains
epochs = 3

# training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    print(f"Number of batches: {num_batches}")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # on encode les entrées en spike trains
        spikes_input = poisson_encoder(images, time_steps)

        state_hidden = model.lif_hidden.initial_state(torch.zeros(images.size(0), 128).to(device))
        state_output = model.lif_output.initial_state(torch.zeros(images.size(0), 10).to(device))

        # forward pass
        spike_outputs = []
        for t in range(time_steps):
            spike_output, state_hidden, state_output = model(
                spikes_input[t], state_hidden, state_output
            )
            spike_outputs.append(spike_output)

        spike_outputs = torch.stack(spike_outputs).mean(dim=0)

        # loss et changement des poids
        loss = loss_fn(spike_outputs, labels)
        print(f"Loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print image index
        print(f"Batch {batch_idx} / {num_batches}")

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        spikes_input = poisson_encoder(images, time_steps)
        state_hidden = model.lif_hidden.initial_state(
            torch.zeros(images.size(0), 128).to(device)
        )
        state_output = model.lif_output.initial_state(
            torch.zeros(images.size(0), 10).to(device)
        )

        spike_outputs = []
        for t in range(time_steps):
            spike_output, state_hidden, state_output = model(
                spikes_input[t], state_hidden, state_output
            )
            spike_outputs.append(spike_output)

        spike_outputs = torch.stack(spike_outputs).mean(dim=0)
        _, predicted = spike_outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
