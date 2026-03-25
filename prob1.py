# =====================================================
# IIT JODHPUR WORD EMBEDDING ASSIGNMENT
# TASK-1 + TASK-2 (FROM SCRATCH, NUMPY ONLY)
# =====================================================

import re
import numpy as np
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# =====================================================
# TASK-1: DATA PREPROCESSING
# =====================================================

# Step 1: Read the merged text file
with open("C:/Users/dell/Downloads/text.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

print("Original text length:", len(corpus))


# Step 2: Clean the text
# Convert to lowercase
corpus = corpus.lower()

# Remove URLs
corpus = re.sub(r"http\S+", "", corpus)

# Remove numbers
corpus = re.sub(r"\d+", "", corpus)

# Remove punctuation (keep only letters)
corpus = re.sub(r"[^a-z\s]", "", corpus)

# Remove extra spaces
corpus = re.sub(r"\s+", " ", corpus).strip()


# Step 3: Tokenization (simple split)
tokens = corpus.split()
print("Total tokens before cleaning:", len(tokens))


# Step 4: Remove stopwords
stopwords = {
    "the","is","and","in","to","of","for","on","with","a","an","by","as",
    "at","from","that","this","it","be","are","was","were","or","which"
}

clean_tokens = [w for w in tokens if w not in stopwords and len(w) > 2]

print("Total tokens after cleaning:", len(clean_tokens))


# Step 5: Dataset statistics
print("\n--- DATASET STATISTICS ---")
print("Documents: 3")
print("Total Tokens:", len(clean_tokens))
print("Vocabulary Size:", len(set(clean_tokens)))


# Step 6: Word Cloud
text = " ".join(clean_tokens)

wc = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.imshow(wc)
plt.axis("off")
plt.title("Word Cloud")
plt.savefig("wordcloud.png")
plt.show()


# =====================================================
# TASK-2: WORD2VEC FROM SCRATCH
# =====================================================

# Step 1: Build vocabulary
vocab = list(set(clean_tokens))
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}

vocab_size = len(vocab)
embedding_dim = 50
window_size = 2

print("\nVocabulary size:", vocab_size)


# Step 2: Generate training data (context-target pairs)
def generate_training_data(tokens, window):
    data = []
    for i, word in enumerate(tokens):
        context = []
        for j in range(i - window, i + window + 1):
            if j != i and 0 <= j < len(tokens):
                context.append(tokens[j])
        data.append((word, context))
    return data

training_data = generate_training_data(clean_tokens, window_size)


# Step 3: Initialize weights
W1 = np.random.rand(vocab_size, embedding_dim)
W2 = np.random.rand(embedding_dim, vocab_size)


# Helper functions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =====================================================
# CBOW TRAINING
# =====================================================

def train_cbow(epochs=3, lr=0.01):
    global W1, W2

    for epoch in range(epochs):
        loss = 0

        for target, context_words in training_data:
            context_indices = [word_to_index[w] for w in context_words]
            target_index = word_to_index[target]

            # Average context embeddings
            h = np.mean(W1[context_indices], axis=0)

            # Output
            u = np.dot(h, W2)
            y_pred = softmax(u)

            # Loss
            loss -= np.log(y_pred[target_index] + 1e-9)

            # Error
            e = y_pred
            e[target_index] -= 1

            # Update W2
            W2 -= lr * np.outer(h, e)

            # Update W1
            for idx in context_indices:
                W1[idx] -= lr * np.dot(W2, e) / len(context_indices)

        print(f"CBOW Epoch {epoch+1}, Loss: {loss:.4f}")


# =====================================================
# SKIP-GRAM WITH NEGATIVE SAMPLING
# =====================================================

def train_skipgram(epochs=3, lr=0.01, neg_samples=5):
    global W1, W2

    for epoch in range(epochs):
        loss = 0

        for target, context_words in training_data:
            target_idx = word_to_index[target]

            for context_word in context_words:
                context_idx = word_to_index[context_word]

                # Positive sample
                score = np.dot(W1[target_idx], W2[:, context_idx])
                prob = sigmoid(score)

                loss -= np.log(prob + 1e-9)
                grad = prob - 1

                W1[target_idx] -= lr * grad * W2[:, context_idx]
                W2[:, context_idx] -= lr * grad * W1[target_idx]

                # Negative samples
                for _ in range(neg_samples):
                    neg_idx = random.randint(0, vocab_size - 1)

                    score = np.dot(W1[target_idx], W2[:, neg_idx])
                    prob = sigmoid(score)

                    loss -= np.log(1 - prob + 1e-9)
                    grad = prob

                    W1[target_idx] -= lr * grad * W2[:, neg_idx]
                    W2[:, neg_idx] -= lr * grad * W1[target_idx]

        print(f"Skip-gram Epoch {epoch+1}, Loss: {loss:.4f}")


# =====================================================
# TRAIN MODELS
# =====================================================

print("\nTraining CBOW...")
train_cbow()

print("\nTraining Skip-gram...")
train_skipgram()


# =====================================================
# SIMILAR WORDS (TESTING)
# =====================================================

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def most_similar(word, top_n=5):
    target_vec = W1[word_to_index[word]]
    similarities = []

    for w in vocab:
        if w != word:
            sim = cosine_similarity(target_vec, W1[word_to_index[w]])
            similarities.append((w, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


print("\nWords similar to 'research':")
print(most_similar("research"))


####task 3 ###
print("\n================ TASK-3: SEMANTIC ANALYSIS ================\n")

# Helper: get top 5 similar words
def print_neighbors(word):
    print(f"\nTop 5 words similar to '{word}':")
    try:
        neighbors = most_similar(word, top_n=5)
        for w, score in neighbors:
            print(f"{w} ({score:.4f})")
    except:
        print("Word not found in vocabulary.")

# Words given in assignment
words_to_check = ["research", "student", "phd", "exam"]

for w in words_to_check:
    print_neighbors(w)


# ---------------------------
# ANALOGY FUNCTION
# ---------------------------

def analogy(word_a, word_b, word_c, top_n=3):
    """
    Solves: A : B :: C : ?
    vector = B - A + C
    """
    try:
        vec = W1[word_to_index[word_b]] - W1[word_to_index[word_a]] + W1[word_to_index[word_c]]
    except:
        print("One of the words not in vocab")
        return

    similarities = []
    for w in vocab:
        if w not in [word_a, word_b, word_c]:
            sim = cosine_similarity(vec, W1[word_to_index[w]])
            similarities.append((w, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


# ---------------------------
# ANALOGY EXPERIMENTS
# ---------------------------

print("\n--- Analogy Experiments ---")

print("\nUG : btech :: PG : ?")
print(analogy("ug", "btech", "pg"))

print("\nstudent : exam :: researcher : ?")
print(analogy("student", "exam", "research"))

print("\nphd : research :: btech : ?")
print(analogy("phd", "research", "btech"))




####task 4 


print("\n================ TASK-4: VISUALIZATION ================\n")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Select some important words to visualize
selected_words = ["research", "student", "phd", "exam", "course", "faculty", "data"]

# Filter words present in vocab
selected_words = [w for w in selected_words if w in word_to_index]

# Get embeddings
embeddings = np.array([W1[word_to_index[w]] for w in selected_words])


# ---------------------------
# PCA Visualization
# ---------------------------

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(8,6))
for i, word in enumerate(selected_words):
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word)

plt.title("PCA Visualization of Word Embeddings")
plt.savefig("pca_plot.png")
plt.show()


# ---------------------------
# t-SNE Visualization
# ---------------------------

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced_tsne = tsne.fit_transform(embeddings)

plt.figure(figsize=(8,6))
for i, word in enumerate(selected_words):
    x, y = reduced_tsne[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word)

plt.title("t-SNE Visualization of Word Embeddings")
plt.savefig("tsne_plot.png")
plt.show()