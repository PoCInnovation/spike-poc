from operator import index
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision.datasets import MNIST 
from torch.utils.data import DataLoader 
from torchvision import transforms  
import norse.torch as snn  
from tqdm import tqdm  
from colorama import init, Fore, Style
import multiprocessing
from torch.amp import autocast, GradScaler

# Initialize colorama
init()  # initialise colorama pour la coloration du terminal
num_workers = multiprocessing.cpu_count()  # définit le nombre de workers égal au nombre de coeurs cpu

# -----------------------------
# Step 1: load le dataset
# -----------------------------

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convertit les images en tenseurs pytorch
        transforms.Normalize((0.5,), (0.5,)),  # Normalise les valeurs des images entre -1 et 1
    ]
)

train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)  # charge le dataset d'entrainement
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)  # charge le dataset de test

train_loader = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True,  # mélange les données à chaque epoch
    num_workers=num_workers,
    pin_memory=True,  # opti
)
test_loader = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


# -----------------------------
# Step 2: on encode en spike train
# -----------------------------
def poisson_encoder(images, time_steps: int):
    batch_size = images.shape[0]
    images = images.view(batch_size, -1)  # flatten les images
    spikes = torch.empty(time_steps, batch_size, images.shape[1], device=images.device)  # cree un tenseur vide pour les spikes.
    spikes.uniform_()  # Remplit le tenseur avec des valeurs uniformes.
    spikes = (spikes < images.unsqueeze(0)).float()  # encode les spikes en fonction des intensités des images
    return spikes


# -----------------------------
# Step 3: creation du SNN
# -----------------------------
class SpikingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Linear(784, 128)  # couche linéaire d'entrée vers cachée
        self.lif_hidden = snn.LIFCell()  # cellule LIF pour la couche cachée.

        self.hidden_to_output = nn.Linear(128, 10)  # couche linéaire de cachée vers sortie
        self.lif_output = snn.LIFCell()  # cellule LIF pour la couche de sortie.

    def forward(self, x, state_hidden=None, state_output=None):
        x = self.input_to_hidden(x)  # Applique la couche d'entrée vers cachée.
        spikes_hidden, state_hidden = self.lif_hidden(x, state_hidden)  # Passe à travers la cellule LIF cachée.

        x = self.hidden_to_output(spikes_hidden)  # applique la couche cachée vers sortie.
        spikes_output, state_output = self.lif_output(x, state_output)  # passe à travers la cellule LIF de sortie.

        return spikes_output, state_hidden, state_output  # Retourne les spikes et les états.


# -----------------------------
# Test du SNN
# -----------------------------

# batch_size = 64
# time_steps = 100
# spike_input = torch.rand(time_steps, batch_size, 784)  # Génère des spike trains aléatoires.

# Properly initialize states
# state_hidden = model.lif_hidden.initial_state(torch.zeros([batch_size, 128]))
# state_output = model.lif_output.initial_state(torch.zeros([batch_size, 10]))

# Forward pass over time
# for t in range(time_steps):
#     spike_output, state_hidden, state_output = model(
#         spike_input[t], state_hidden, state_output
#     )

# print("Output shape:", spike_output.shape)


# initialisation
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps") # pour mac
    if torch.cuda.is_available():
        return torch.device("cuda")  # pour gpu
    return torch.device("cpu")  # pour cpu


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    model = SpikingNN().to(device)  # initialise le modèle
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.1
    )  # ???
    loss_fn = nn.MSELoss()
    scaler = GradScaler()  # ???

    time_steps = 100  # Durée des spike trains.
    epochs = 20

    class EarlyStopping:
        def __init__(self, patience=7, min_delta=0):
            self.patience = patience  # temps avant arrêt.
            self.min_delta = min_delta  # delta minimal pour une amélioration
            self.counter = 0  # compteur de non améliorations
            self.best_loss = None
            self.early_stop = False # si a true, arrête l'entraînement

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss 
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss 
                self.counter = 0 

    # training loop
    for epoch in range(epochs):
        model.train()  # met le modèle en mode entraînement
        total_loss = 0  # initialise la perte par epoch
        num_batches = len(train_loader)  # nombre de batches par epoch

        with tqdm(
            train_loader, desc=f"{Fore.CYAN}Epoch {epoch+1}/{epochs}{Style.RESET_ALL}"
        ) as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()  # reinitialise les gradients

                with autocast("cuda"):
                    # on encode les entrées en spike trains
                    spikes_input = poisson_encoder(images, time_steps)  # Encode les images en spike trains.

                    state_hidden = model.lif_hidden.initial_state(
                        torch.zeros(images.size(0), 128).to(device)
                    )  # initialise le hidden state
                    state_output = model.lif_output.initial_state(
                        torch.zeros(images.size(0), 10).to(device)
                    )  # Initialise l'output state

                    # forward pass
                    spike_outputs = []
                    for t in range(time_steps):
                        spike_output, state_hidden, state_output = model(
                            spikes_input[t], state_hidden, state_output
                        )  # passe les spikes à travers le modèle
                        spike_outputs.append(spike_output)  # Ajoute la sortie aux spikes.

                    spike_outputs = torch.stack(spike_outputs).mean(dim=0)  # Moyenne les sorties sur le temps.

                    # convert en one hot pour le mse
                    target_one_hot = torch.zeros(labels.size(0), 10, device=device)  # cree un tensor cible.
                    target_one_hot.scatter_(1, labels.unsqueeze(1), 1)  # Convertit les labels en one-hot.

                    # loss et changement des poids
                    loss = loss_fn(spike_outputs, target_one_hot)  # calcule la perte

                scaler.scale(loss).backward()  # backpropagation avec scaling
                scaler.step(optimizer)  # met à jour les poids
                scaler.update()  # met à jour le scaler

                total_loss += loss.item()  # accumule la perte
                pbar.set_postfix(
                    {"loss": f"{Fore.YELLOW}{loss.item():.4f}{Style.RESET_ALL}"}
                )  # affiche la perte dans la barre de progression.

            avg_loss = total_loss / len(train_loader)  # calcule la perte moyenne.
            print(
                f"{Fore.GREEN}Epoch {epoch+1} - Average Loss: {avg_loss:.4f}{Style.RESET_ALL}"
            )  # affiche la perte moyenne.

        scheduler.step(total_loss / len(train_loader))  # met à jour le scheduler.

    model.eval() 
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

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

    print(
        f"{Fore.GREEN}Test Accuracy: {Fore.YELLOW}{100 * correct / total:.2f}%{Style.RESET_ALL}"
    )
