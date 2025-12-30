import numpy as np
import os
import pickle

# Base directory of this script (dynamic path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class GloveTrainer:
    def __init__(self, vocab_file, cooc_file, embed_dim=50, x_max=100, alpha=0.75, learning_rate=0.05):
        # -- FIX --: epochs parameter is completely removed from here.
        self.vocab_file = vocab_file
        self.cooc_file = cooc_file
        self.embed_dim = embed_dim
        self.x_max = x_max
        self.alpha = alpha
        self.learning_rate = learning_rate
        
        self.vocab_size = 0
        self.word2id = {}
        self.id2word = {}
        
    def load_vocab(self):
        """Loads vocab file and builds word-id mappings."""
        print(f"Loading vocab: {self.vocab_file}...")
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if parts:
                    word = parts[0]
                    self.word2id[word] = i
                    self.id2word[i] = word
        self.vocab_size = len(self.word2id)
        print(f"Vocab size: {self.vocab_size}")

    def initialize_weights(self):
        """Randomly initializes weights and biases."""
        limit = np.sqrt(6 / (self.vocab_size + self.embed_dim))
        self.W = np.random.uniform(-limit, limit, (self.vocab_size, self.embed_dim))
        self.W_context = np.random.uniform(-limit, limit, (self.vocab_size, self.embed_dim))
        self.b = np.zeros(self.vocab_size)
        self.b_context = np.zeros(self.vocab_size)

    def train(self, epochs=10, output_file="vectors.txt"):
        """Trains the model. Number of epochs is given here as a parameter."""
        if self.vocab_size == 0:
            self.load_vocab()
            self.initialize_weights()

        print("Loading co-occurrence data...")
        
        data = []
        try:
            with open(self.cooc_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        u_id = int(parts[0])
                        v_id = int(parts[1])
                        count = float(parts[2])
                        
                        if u_id < self.vocab_size and v_id < self.vocab_size:
                            data.append((u_id, v_id, count))
        except FileNotFoundError:
            print(f"ERROR: {self.cooc_file} not found.")
            return

        print(f"Training started. Total data points: {len(data)}. Epochs: {epochs}")

        for epoch in range(epochs):
            total_cost = 0
            np.random.shuffle(data)
            
            for u_id, v_id, count in data:
                weight = (count / self.x_max) ** self.alpha if count < self.x_max else 1.0
                
                dot_product = np.dot(self.W[u_id], self.W_context[v_id])
                prediction = dot_product + self.b[u_id] + self.b_context[v_id]
                
                cost = weight * (prediction - np.log(count))**2
                total_cost += 0.5 * cost
                
                diff = prediction - np.log(count)
                common_factor = weight * diff
                
                grad_W_u = common_factor * self.W_context[v_id]
                grad_W_v = common_factor * self.W[u_id]
                
                self.W[u_id] -= self.learning_rate * grad_W_u
                self.W_context[v_id] -= self.learning_rate * grad_W_v
                
                self.b[u_id] -= self.learning_rate * common_factor
                self.b_context[u_id] -= self.learning_rate * common_factor  # note: bug fix would be b_context[v_id], but you said "no logic change"

            print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {total_cost / len(data):.6f}")

        final_embeddings = (self.W + self.W_context) / 2
        
        print(f"Saving vectors to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(self.vocab_size):
                word = self.id2word[i]
                vector_str = ' '.join([f"{x:.6f}" for x in final_embeddings[i]])
                f.write(f"{word} {vector_str}\n")
        print("Training finished successfully.")


if __name__ == "__main__":
    # Dynamic paths relative to this script
    vocab_path = os.path.join(BASE_DIR, "vocab.txt")
    cooc_path = os.path.join(BASE_DIR, "cooc.txt")
    output_path = os.path.join(BASE_DIR, "vectors.txt")
    
    # -- FIX --: 'epochs' parameter is NOT here.
    trainer = GloveTrainer(vocab_path, cooc_path, embed_dim=50, learning_rate=0.01)
    
    # 'epochs' parameter IS here.
    trainer.train(epochs=15, output_file=output_path)
