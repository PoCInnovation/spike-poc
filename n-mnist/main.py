import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from colorama import init, Fore, Style
import multiprocessing
from collections import namedtuple

# Initialize colorama
init()

# LIF Neuron State holder
LIFState = namedtuple("LIFState", ["v", "i"])


class LIFCell:
    def __init__(self, tau_mem=20.0, v_threshold=1.0):
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold

    def initial_state(self, batch_size, n_neurons):
        return LIFState(
            v=np.zeros((batch_size, n_neurons)), i=np.zeros((batch_size, n_neurons))
        )

    def forward(self, x, state):
        # Compute new membrane potential
        dv = (x - state.v) / self.tau_mem
        v_new = state.v + dv

        # Generate spikes
        spikes = (v_new >= self.v_threshold).astype(float)

        # Reset membrane potential where spikes occurred
        v_new *= 1 - spikes

        return spikes, LIFState(v=v_new, i=x)


class SpikingNN:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01

        self.lif_hidden = LIFCell()
        self.lif_output = LIFCell()

    def forward(self, x, state_hidden=None, state_output=None):
        batch_size = x.shape[0]

        if state_hidden is None:
            state_hidden = self.lif_hidden.initial_state(batch_size, 128)
        if state_output is None:
            state_output = self.lif_output.initial_state(batch_size, 10)

        # Hidden layer
        hidden = np.dot(x, self.weights1)
        spikes_hidden, state_hidden = self.lif_hidden.forward(hidden, state_hidden)

        # Output layer
        output = np.dot(spikes_hidden, self.weights2)
        spikes_output, state_output = self.lif_output.forward(output, state_output)

        return spikes_output, state_hidden, state_output


def poisson_encoder(images, time_steps):
    batch_size = images.shape[0]
    flat_images = images.reshape(batch_size, -1)
    spikes = np.random.random((time_steps, batch_size, flat_images.shape[1]))
    return (spikes < flat_images).astype(float)


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    # Load MNIST with PyTorch
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

    # Convert to numpy arrays
    X_train = to_numpy(train_dataset.data).reshape(-1, 784) / 255.0
    y_train = to_numpy(train_dataset.targets)
    X_test = to_numpy(test_dataset.data).reshape(-1, 784) / 255.0
    y_test = to_numpy(test_dataset.targets)

    # Initialize model and parameters
    model = SpikingNN()
    learning_rate = 0.001
    time_steps = 100
    batch_size = 100
    epochs = 20

    # Training loop
    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        total_loss = 0

        with tqdm(
            range(n_batches),
            desc=f"{Fore.CYAN}Epoch {epoch+1}/{epochs}{Style.RESET_ALL}",
        ) as pbar:
            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_x = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]

                # Create one-hot encoded targets
                targets = np.zeros((batch_size, 10))
                targets[np.arange(batch_size), batch_y] = 1

                # Forward pass
                spikes_input = poisson_encoder(batch_x, time_steps)
                spike_outputs = []

                state_hidden = None
                state_output = None

                for t in range(time_steps):
                    spike_output, state_hidden, state_output = model.forward(
                        spikes_input[t], state_hidden, state_output
                    )
                    spike_outputs.append(spike_output)

                spike_outputs = np.mean(np.stack(spike_outputs), axis=0)

                # Compute loss (MSE)
                loss = np.mean((spike_outputs - targets) ** 2)
                total_loss += loss

                # Simple gradient descent update
                error = spike_outputs - targets

                # Update weights (simplified backprop)
                model.weights2 -= learning_rate * np.dot(state_hidden.v.T, error)
                hidden_error = np.dot(error, model.weights2.T)
                model.weights1 -= learning_rate * np.dot(batch_x.T, hidden_error)

                pbar.set_postfix({"loss": f"{Fore.YELLOW}{loss:.4f}{Style.RESET_ALL}"})

        avg_loss = total_loss / n_batches
        print(
            f"{Fore.GREEN}Epoch {epoch+1} - Average Loss: {avg_loss:.4f}{Style.RESET_ALL}"
        )

    # Testing
    correct = 0
    total = 0
    n_test_batches = len(X_test) // batch_size

    for batch_idx in range(n_test_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        batch_x = X_test[start_idx:end_idx]
        batch_y = y_test[start_idx:end_idx]

        spikes_input = poisson_encoder(batch_x, time_steps)
        spike_outputs = []
        state_hidden = None
        state_output = None

        for t in range(time_steps):
            spike_output, state_hidden, state_output = model.forward(
                spikes_input[t], state_hidden, state_output
            )
            spike_outputs.append(spike_output)

        spike_outputs = np.mean(np.stack(spike_outputs), axis=0)
        predictions = np.argmax(spike_outputs, axis=1)

        correct += np.sum(predictions == batch_y)
        total += batch_size

    accuracy = 100 * correct / total
    print(f"{Fore.GREEN}Test Accuracy: {Fore.YELLOW}{accuracy:.2f}%{Style.RESET_ALL}")
