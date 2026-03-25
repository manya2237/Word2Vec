# name - manya , roll number - b22cs032 
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
# device setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# LOAD DATA

def load_data(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print("File not found:", file_path)
        exit()

    # Read names, remove empty lines, convert to lowercase
    with open(file_path, "r", encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names

# Load dataset
names = load_data(r"C:\Users\dell\Downloads\TrainingNames.txt")



# VOCAB CREATION

# Extract all unique characters from dataset
cha = sorted(list(set("".join(names))))

# Add special tokens:
# <PAD> = padding, <SOS> = start, <EOS> = end
cha = ['<PAD>', '<SOS>', '<EOS>'] + cha

# Create mappings
char2idx = {ch: i for i, ch in enumerate(cha)}   # char → index
idx2char = {i: ch for ch, i in char2idx.items()}   # index → char

v_size = len(cha)


# ENCODING

def encode(name):
    
 #   Convert name into sequence of indices
 #   Example: "ram" → [SOS, r, a, m, EOS]
    
    return [char2idx['<SOS>']] + [char2idx[c] for c in name] + [char2idx['<EOS>']]

# Encode all names
encoded = [encode(n) for n in names]



# BATCHING

def pad(seq, max_len):
    """Pad sequence to same length using <PAD>"""
    return seq + [char2idx['<PAD>']] * (max_len - len(seq))


def get_batch(data, batch_size=32):
    """
    Create training batch:
    X = input sequence
    Y = target sequence (shifted by 1)
    """
    batch = random.sample(data, batch_size)

    # Find longest sequence in batch
    max_len = max(len(x) for x in batch)

    X, Y = [], []
    for seq in batch:
        seq = pad(seq, max_len)

        # Input = all except last
        # Target = all except first
        X.append(seq[:-1])
        Y.append(seq[1:])

    return torch.tensor(X).to(device), torch.tensor(Y).to(device)



# MODEL 1: VANILLA RNN

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()

        # Convert character index → embedding vector
        self.embed = nn.Embedding(vocab_size, hidden)

        # Basic RNN layer
        self.rnn = nn.RNN(hidden, hidden, batch_first=True)

        # Final layer → predicts next character
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embed(x)           # (batch, seq_len, hidden)
        out, _ = self.rnn(x)        # pass through RNN
        return self.fc(out)         # output logits


# MODEL 2: BLSTM

class CharBLSTM(nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden)

        # Bidirectional LSTM (forward + backward)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)

        # Output size is doubled because of bidirectional
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out)



# MODEL 3: ATTENTION RNN

class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.RNN(hidden, hidden, batch_first=True)

        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)

        # SELF-ATTENTION
        # Compare each position with every other position
        sc = torch.bmm(out, out.transpose(1, 2))

        # Convert scores → probabilities
        attn_weights = torch.softmax(sc, dim=-1)

        # Weighted combination of hidden states
        context = torch.bmm(attn_weights, out)

        return self.fc(context)



# TRAINING FUNCTION

def train(model, epochs=30, lr=0.003):
    model.to(device)

    opti = optim.Adam(model.parameters(), lr=lr)

    # Ignore PAD tokens in loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=char2idx['<PAD>'])

    for ep in range(epochs):
        model.train()

        X, Y = get_batch(encoded)

        out = model(X)

        # Flatten for loss calculation
        loss = loss_fn(out.view(-1, v_size), Y.view(-1))

        opti.zero_grad()
        loss.backward()
        opti.step()

        if ep % 5 == 0:
            print(f"Epoch {ep} | Loss: {loss.item():.4f}")

    return model



# NAME GENERATION

def generate(model, max_len=15, temperature=0.8):
    model.eval()

    # Start with <SOS>
    inp = torch.tensor([[char2idx['<SOS>']]]).to(device)
    name = ""

    for _ in range(max_len):
        out = model(inp)

        # Apply temperature (controls randomness)
        probs = torch.softmax(out[:, -1, :] / temperature, dim=-1)

        # Sample next character
        while True:
            idx = torch.multinomial(probs, 1).item()
            ch = idx2char[idx]

            # Avoid invalid tokens
            if ch not in ['<PAD>', '<SOS>']:
                break

        # Stop if end token
        if ch == '<EOS>':
            break

        name += ch

        # Append new character to input
        inp = torch.cat([inp, torch.tensor([[idx]]).to(device)], dim=1)

    return name


def generate_many(model, n=200):
    """Generate multiple names"""
    return [generate(model) for _ in range(n)]



# EVALUATION

def evaluate(gen, train):
    gen_set = set(gen)
    train_set = set(train)

    # Names not present in training set
    novel = [x for x in gen if x not in train_set]

    nove = len(novel) / len(gen) * 100
    di = len(gen_set) / len(gen) * 100

    return nove, di



# PARAMETER COUNT

def count_params(m):
    """Count total trainable parameters"""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)



# MAIN EXECUTION

hidden = 128

print("\n--- Training RNN ---")
rnn = train(CharRNN(v_size, hidden))

print("\n--- Training BLSTM ---")
blstm = train(CharBLSTM(v_size, hidden))

print("\n--- Training Attention ---")
attn = train(AttentionRNN(v_size, hidden))



# GENERATION

rnn_gen = generate_many(rnn)
blstm_gen = generate_many(blstm)
attn_gen = generate_many(attn)



# EVALUATION

rnn_nov, rnn_div = evaluate(rnn_gen, names)
blstm_nov, blstm_div = evaluate(blstm_gen, names)
attn_nov, attn_div = evaluate(attn_gen, names)

def model_stats(model):
    """
    Returns:
    - number of trainable parameters
    - model size in MB (assuming 32-bit = 4 bytes per parameter)
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = params * 4 / (1024 * 1024)
    return params, size_mb

print("\nRESULTS ")

print("\nRNN")
print("Params:", count_params(rnn))
print(f"Novelty: {rnn_nov:.2f}%")
print(f"Diversity: {rnn_div:.2f}%")

print("\nBLSTM")
print("Params:", count_params(blstm))
print(f"Novelty: {blstm_nov:.2f}%")
print(f"Diversity: {blstm_div:.2f}%")

print("\nAttention")
print("Params:", count_params(attn))
print(f"Novelty: {attn_nov:.2f}%")
print(f"Diversity: {attn_div:.2f}%")



# SAMPLE OUTPUT

print("\nSample Names:\n")
print("RNN:", rnn_gen[:10])
print("BLSTM:", blstm_gen[:10])
print("Attention:", attn_gen[:10])