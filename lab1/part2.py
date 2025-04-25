import os
import numpy as np
import comet_ml
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write
import mitdeeplearning as mdl

# Set your Comet API key
COMET_API_KEY = ""

# Ensure GPU is available and API key is set
assert torch.cuda.is_available(), "Please enable GPU from runtime settings"
assert COMET_API_KEY != "", "Please insert your Comet API Key"

# Load the dataset
songs = mdl.lab1.load_training_data()
example_song = songs[0]
print("\nExample song: ")
print(example_song)

# Prepare text data
songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))
print(f"There are {len(vocab)} unique characters in the dataset")

# Create mappings from characters to indices and back
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Vectorization function
def vectorize_string(string):
    """
    Convert a string to a numpy array of indices using char2idx mapping.
    """
    return np.array([char2idx[c] for c in string])

# Vectorize all songs
vectorized_songs = vectorize_string(songs_joined)

# Batch creation function
def get_batch(vectorized_songs, seq_length, batch_size):
    """
    Create input and target sequences for training.
    """
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n - seq_length, batch_size)
    input_batch = [vectorized_songs[i:i+seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1:i+seq_length+1] for i in idx]
    x_batch = torch.tensor(input_batch, dtype=torch.long)
    y_batch = torch.tensor(output_batch, dtype=torch.long)
    return x_batch, y_batch

# Test batch function
test_args = (vectorized_songs, 10, 2)
x_batch, y_batch = get_batch(*test_args)
assert x_batch.shape == (2, 10), "x_batch shape is incorrect"
assert y_batch.shape == (2, 10), "y_batch shape is incorrect"
print("Batch function works correctly!")

# Define the LSTM-based RNN model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)
        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)
        out = self.fc(out)
        return out if not return_state else (out, state)

# Loss function and computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cross_entropy = nn.CrossEntropyLoss()

def compute_loss(labels, logits):
    # Flatten labels to (B*L,)
    batched_labels = labels.view(-1)
    # Flatten logits to (B*L, V)
    batched_logits = logits.view(-1, logits.size(2))
    loss = cross_entropy(batched_logits, batched_labels)
    return loss

# Test compute_loss with untrained model
vocab_size = len(vocab)
embedding_dim = 256
hidden_size = 1024
model = LSTMModel(vocab_size, embedding_dim, hidden_size).to(device)
y_batch = y_batch.to(device)
x_batch = x_batch.to(device)
pred = model(x_batch)  # (batch, seq_length, vocab_size)
example_batch_loss = compute_loss(y_batch, pred)
print(f"Prediction shape: {pred.shape}")
print(f"scalar_loss:      {example_batch_loss.item()}")

# Hyperparameters and checkpoints
params = dict(
    num_training_iterations=3000,
    batch_size=8,
    seq_length=100,
    learning_rate=5e-3,
    embedding_dim=256,
    hidden_size=1024,
)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

# Setup Comet experiment
def create_experiment():
    if 'experiment' in globals():
        experiment.end()
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name="6S191_Lab1_Part2"
    )
    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()
    return experiment

experiment = create_experiment()

# Instantiate model and optimizer for training
model = LSTMModel(vocab_size, params['embedding_dim'], params['hidden_size']).to(device)
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

# Training loop
history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
if hasattr(tqdm, '_instances'):
    tqdm._instances.clear()
for iter in tqdm(range(params['num_training_iterations'])):
    x_batch, y_batch = get_batch(vectorized_songs, params['seq_length'], params['batch_size'])
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    # Forward and backward pass
    model.train()
    optimizer.zero_grad()
    y_hat = model(x_batch)
    loss = compute_loss(y_batch, y_hat)
    loss.backward()
    optimizer.step()
    # Log and plot
    experiment.log_metric("loss", loss.item(), step=iter)
    history.append(loss.item())
    plotter.plot(history)
    # Save checkpoint
    if iter % 100 == 0:
        torch.save(model.state_dict(), checkpoint_prefix)
# Save final model
torch.save(model.state_dict(), checkpoint_prefix)
experiment.flush()

# Function to generate music

def generate_text(model, start_string, generation_length=1000):
    # Convert start string to indices
    input_indices = [char2idx[c] for c in start_string]
    input_idx = torch.tensor([input_indices], dtype=torch.long).to(device)
    state = model.init_hidden(input_idx.size(0), device)
    text_generated = []
    tqdm._instances.clear()

    for _ in tqdm(range(generation_length)):
        predictions, state = model(input_idx, state, return_state=True)
        predictions = predictions.squeeze(0)[-1]  # last timestep
        # Sample from the distribution
        probabilities = torch.softmax(predictions, dim=-1)
        next_idx = torch.multinomial(probabilities, num_samples=1).item()
        # Add predicted character
        text_generated.append(idx2char[next_idx])
        # Prepare next input
        input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return start_string + ''.join(text_generated)

# Generate and play songs
generated_songs = []
seed = "X"
generated_song = generate_text(model, seed, 1000)
generated_songs.append(generated_song)

# Synthesize and save
for i, song in enumerate(generated_songs):
    waveform = mdl.lab1.play_song(song)
    if waveform:
        print("Generated song", i)
        ipythondisplay.display(waveform)
        numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
        wav_file_path = f"output_{i}.wav"
        write(wav_file_path, 88200, numeric_data)
        experiment.log_asset(wav_file_path)
