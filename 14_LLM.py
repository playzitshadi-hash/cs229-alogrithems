from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
import random
import torch.nn as nn
import torch.optim as optim
class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.fc1 = nn.Linear(embedding_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self,x):
        x=  self.embedding(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self, embda=64, head=16, embed_dim=None, num_heads=None):
        super().__init__()
        
        if embed_dim is not None:
            embda = embed_dim
        if num_heads is not None:
            head = num_heads
            
        self.embda_dim = embda
        self.num_heads = head
        self.head_dim = self.embda_dim // self.num_heads
        
        self.qkv_proj = nn.Linear(embda, embda * 3)
        self.out_proj = nn.Linear(embda, embda)
        
    def forward(self, X):
        B, T, C = X.shape
        
        qkv = self.qkv_proj(X)
        Q, K, V = qkv.chunk(3, dim=-1)
        
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attention = Q @ K.transpose(-2, -1)
        scale = attention / torch.sqrt(torch.tensor(self.head_dim, dtype=X.dtype, device=X.device))
        
        mask = torch.tril(torch.ones(T, T, device=X.device)).view(1, 1, T, T)
        addition = scale.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(addition, dim=-1)
        Final = attention_weights @ V

        Final = Final.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(Final)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embda=64, head=16):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim=embda, num_heads=head)
        self.ln1 = nn.LayerNorm(embda)
        self.ln2 = nn.LayerNorm(embda)
        self.ffn = nn.Sequential(
            nn.Linear(embda, 4 * embda),
            nn.ReLU(),
            nn.Linear(4 * embda, embda)
        )
        
    def forward(self, X):
        attention_out = self.attention(self.ln1(X))
        X = X + attention_out
        
        ffn_out = self.ffn(self.ln2(X))
        X = X + ffn_out
        return X

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embda, block_size, num_layers, head):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embda)
        self.position_embedding = nn.Embedding(block_size, embda)
        self.blocks = nn.ModuleList([TransformerBlock(embda, head) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embda)
        self.head = nn.Linear(embda, vocab_size)
        
    def forward(self, X):
        B, T = X.shape
        tok_emb = self.embedding(X)
        positions = torch.arange(T, device=X.device)
        pos_emb = self.position_embedding(positions)
        X = tok_emb + pos_emb
        for block in self.blocks:
            X = block(X)
        X = self.ln_f(X)
        logits = self.head(X)
        return logits



sentence=  "Harvard provides financial aid to eligible students."
# unique = sorted(set(sentence))
# char_to_id = {char: i for i, char in enumerate(unique)}
# id_to_char = {i:char for char,i in char_to_id.items()}
# encoded = [char_to_id[c] for c in sentence]
# # print(char_to_id)
# decoded = "".join([id_to_char[i] for i in encoded])
# # print(decoded)
# # print(encoded)
# text = "low low low lower lower newest newest newest"
# unique1 = sorted(set(text))
# # print(unique1)
# corpus = {
#     "l o w </w>": 3,
#     "l o w e r </w>": 2,
#     "n e w e s t </w>": 3,
# }
# pairs=defaultdict(int)
# for iteration in range(5):

#     pairs = defaultdict(int)
#     for word, freq in corpus.items():
#         symbols = word.split()
#         for i in range(len(symbols) - 1):
#             pairs[(symbols[i], symbols[i+1])] += freq

#     best_pair = max(pairs, key=pairs.get)
#     print(f"Iteration {iteration+1}: Merging {best_pair} (Freq: {pairs[best_pair]})")

#     new_corpus = {}
#     for word, freq in corpus.items():
#         symbols = word.split()
#         new_symbols = []
#         i = 0
#         while i < len(symbols):
#             if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i+1] == best_pair[1]:
#                 new_symbols.append(best_pair[0] + best_pair[1])
#                 i += 2
#             else:
#                 new_symbols.append(symbols[i])
#                 i += 1
#         new_word = " ".join(new_symbols)
#         new_corpus[new_word] = freq
#     corpus = new_corpus
# print("\nFinal Vocabulary:")
# print(corpus)
# list = ["lowest </w>" ,"newest </w>","learning</w>","low</w>"]
# class BPETokenizer:
#     def __init__(self):
#         self.merges = []
#         self.vocab = []

#     # Make sure num_merges is included as a parameter here!
#     def train(self, corpus, num_merges=5):
#         # Initial corpus setup: split text into words and count frequencies
#         words = corpus.split()
#         vocab_freqs = {}
#         for word in words:
#             # Represent each word as a space-separated string of characters
#             symbol_word = " ".join(list(word))
#             vocab_freqs[symbol_word] = vocab_freqs.get(symbol_word, 0) + 1

#         # Level 1: The outer loop for the specified number of iterations
#         for iteration in range(num_merges):
            
#             # Level 2: Count the pairs
#             pairs = {}
#             for word, freq in vocab_freqs.items():
#                 symbols = word.split()
#                 for j in range(len(symbols) - 1):
#                     pair = (symbols[j], symbols[j+1])
#                     pairs[pair] = pairs.get(pair, 0) + freq
            
#             if not pairs:
#                 break
                
#             # Level 3: Find the best pair
#             best_pair = max(pairs, key=pairs.get)
            
#             # Level 4: The Merge (using your sliding/checking pointer logic)
#             new_vocab_freqs = {}
#             for word, freq in vocab_freqs.items():
#                 symbols = word.split()
#                 new_symbols = []
#                 i = 0
                
#                 while i < len(symbols):
#                     if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i+1] == best_pair[1]:
#                         new_symbols.append(best_pair[0] + best_pair[1])
#                         i += 2
#                     else:
#                         new_symbols.append(symbols[i])
#                         i += 1
                        
#                 new_word = " ".join(new_symbols)
#                 new_vocab_freqs[new_word] = freq
                
#             vocab_freqs = new_vocab_freqs
            
#             # Level 5: Save the rule
#             self.merges.append(best_pair)
#             print(f"Iteration {iteration + 1}: Merged {best_pair}")

# # --- Example Testing ---
# corpus = "low lower lowest newer wider low lower lowest"
# tokenizer = BPETokenizer()
# tokenizer.train(corpus, 5)

# print("\nSaved Merges:", tokenizer.merges)
# corpus = "machine learning makes machines intelligent. machine learning is fun. python is fun."
# words = corpus.split()
# pairs = defaultdict(int)
# for i in range(len(words)-1):
#     current_words = words[i]
#     next_words = words[i+1]
#     pairs[(current_words,next_words)]+=1
# print(pairs)

# h= []
# followers = {}
# for (word1, word2), count in pairs.items():
#     if word1 not in followers:
#         followers[word1] = []
#     followers[word1].extend([word2] * count)

# print(followers)

# # 2. Set the seed word
# current_word = "makes" 
# h = [current_word] # Start the list with the seed word

# # 3. The generation loop
# for i in range(20):
#     if current_word not in followers:
#         break
    
#     possible_next_words = followers[current_word]
#     next_word = random.choice(possible_next_words)
    
#     h.append(next_word)
#     current_word = next_word

# print(" ".join(h))
# para = "the cat sat on the mat. the cat ate the food. the dog sat on the mat."
# unique = sorted(set(para))
# char_to_id = {char: i for i, char in enumerate(unique)}
# id_to_char = {i:char for char,i in char_to_id.items()}
# vocab_size= len(unique)
# text_ids = torch.tensor([char_to_id[c] for c in para], dtype=torch.long)
# x_tensor=text_ids[:-1]
# y_tensor=text_ids[1:]

# model = TinyLanguageModel(vocab_size, embedding_dim=16, hidden_dim=32)
# criterian=nn.CrossEntropyLoss()
# optimizor = optim.Adam(model.parameters(),lr=0.01)
# for epoch in range(100):
#     predicted = model(x_tensor)
#     loss = criterian(predicted,y_tensor)
#     optimizor.zero_grad()
#     loss.backward()
#     optimizor.step()
#     if epoch % 20 == 0:
#         acc = (torch.argmax(predicted, dim=1) == y_tensor).float().mean().item()
#         print(f"Probe Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")
# prompt = "the c"
# generated = list(prompt)

# # 1. Convert prompt to a tensor
# input_seq = torch.tensor([char_to_id[c] for c in generated], dtype=torch.long)

# model.eval()
# with torch.no_grad():
#     for _ in range(50):
#         # 2. Pass the CURRENT input_seq to the model
#         logits = model(input_seq)
        
#         # 3. Get the prediction for the very last character
#         last_logits = logits[-1]
        
#         # 4. Find the highest score
#         highest_index = torch.argmax(last_logits).item()
        
#         # 5. Convert back to char and append
#         next_char = id_to_char[highest_index]
#         generated.append(next_char)
        
#         # 6. Update the input sequence for the next loop
#         new_input = torch.tensor([highest_index], dtype=torch.long)
#         input_seq = torch.cat((input_seq, new_input))

def softmax(X,axis=-1):
     exp_x = np.exp(X - np.max(X, axis=axis, keepdims=True))
     return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
# print("".join(generated))
# mask = np.zeros(4,4)
# upper_indices = np.triu_indices(4,k=1)
# mask[upper_indices]= -np.inf
# np.random.seed(42)
# X = np.random.randn(4,8)
# Wq = np.random.randn(8,8)
# Wk=np.random.randn(8,8)
# Wv= np.random.randn(8,8)
# Q= X @ Wq
# K = X @ Wk
# V = X @ Wv
# Attention = Q @ K.T
# scale=  Attention/np.sqrt(8)
# addition= scale + mask
# attention_weights = softmax(addition)
# Final  =attention_weights @ V
# print("Q shape:", Q.shape)
# print("Attention weights:\n", attention_weights)
# print("Output shape:", Final.shape)


sentencess = [
    "Artificial intelligence encompasses machine learning models that can process vast amounts of unstructured data to recognize complex patterns.",
    "Large language models leverage transformer architectures with attention mechanisms to generate human-like text responses.",
    " Deep learning utilizes multi-layered artificial neural networks inspired by biological synaptic connections in the human brain.",
    "Computer vision systems empower machines to interpret, analyze, and extract meaningful insights from digital images and videos.",
    "Reinforcement learning trains autonomous agents to make sequences of decisions by rewarding desired behaviors in dynamic environments.",
    "Generative adversarial networks pit two competing neural networks against each other to create highly realistic synthetic media.",
    "Natural language processing bridges the communication gap between human speech patterns and digital machine execution.",
    " Edge AI deployments execute complex inference tasks directly on local hardware devices rather than relying on remote cloud servers.",
    "Transfer learning allows pre-trained neural networks to adapt rapidly to new, specialized tasks with minimal custom training data.",
    "Ethics in artificial intelligence remain paramount as engineers tackle issues of bias, transparency, and data privacy.",

    "Quantum mechanics reveals that subatomic particles can exist in multiple superposition states until they are observed.",
    "The theory of general relativity describes gravity not as a direct force, but as the curvature of spacetime caused by mass and energy.",
    " CRISPR-Cas9 gene editing technology enables precise, targeted modifications to living organism DNA sequences.",
    "Renewable energy integration through solar, wind, and advanced battery storage is transforming global power grids.",
    "Plate tectonics drives the continuous reshaping of Earth's crust, causing earthquakes, volcanic activity, and mountain formation.",
    "Astrophysics studies celestial phenomena ranging from the expansion of the universe to the extreme gravitational pull of black holes.",
    "Cellular respiration converts biochemical energy from nutrients into adenosine triphosphate to fuel biological processes.",
    " The James Webb Space Telescope peers deep into the infrared spectrum to capture light from the earliest galaxies formed after the Big Bang.",
    "Epigenetics explores how environmental factors and behavioral choices can alter gene expression without modifying underlying DNA sequences.",
    "Thermodynamics dictates that total entropy in an isolated system will always increase over time, driving the arrow of time.",

    "Object-oriented programming structures software around reusable classes and instances that encapsulate both data and behavior.",
    "Version control systems like Git track incremental code changes across teams, allowing developers to collaborate seamlessly without overwriting work.",
    " Asynchronous programming models use event loops and non-blocking I/O operations to handle concurrent tasks efficiently without freezing applications.",
    "Algorithm time and space complexity analysis helps engineers evaluate how resource demands scale as input sizes grow larger.",
    "Functional programming emphasizes immutability, pure functions, and the avoidance of shared state to write more predictable code.",
    "Relational databases organize information into structured tables with primary and foreign keys to maintain rigorous data integrity.",
    "Application programming interfaces establish standard communication contracts that allow disparate software systems to interact securely.",
    " Test-driven development requires writing automated unit tests before implementing the actual functional code to ensure robust coverage.",
    "Memory management in modern languages often relies on automated garbage collection to reclaim unused heap allocations safely.",
    "Containerization technologies package applications and their dependencies into standardized units to ensure consistent execution across any environment.",
]

full_text = " ".join(sentencess)
unique_chars = sorted(set(full_text))
vocab_size = len(unique_chars)
char_to_id = {char: i for i, char in enumerate(unique_chars)}
id_to_char = {i: char for char, i in char_to_id.items()}

print(f"Vocab size: {vocab_size} unique characters.")
print(f"Total text length: {len(full_text)} characters.")

# --- Data Setup ---
text_ids = [char_to_id[c] for c in full_text]
text_tensor_full = torch.tensor(text_ids, dtype=torch.long)

x_tensor = text_tensor_full[:-1].unsqueeze(0)  # Shape: (1, N-1)
y_tensor = text_tensor_full[1:].unsqueeze(0)   # Shape: (1, N-1)

# Set block_size to match the length of the sequence (or use chunks)
block_size = x_tensor.size(1)

# --- Initialize Model, Loss, & Optimizer ---
model = TinyTransformer(vocab_size=vocab_size, embda=16, block_size=block_size, num_layers=1, head=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Training Loop ---
print("\nStarting Training...")
for epoch in range(100):
    predicted = model(x_tensor)  # Shape: (1, T, vocab_size)
    
    predicted_flat = predicted.view(-1, vocab_size)
    y_flat = y_tensor.view(-1)
    
    loss = criterion(predicted_flat, y_flat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        preds = torch.argmax(predicted, dim=-1)
        acc = (preds == y_tensor).float().mean().item()
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")

print("\nTraining complete!")
temperature = 0.7
generated_text = "Artificial intelligence"

model.eval()
with torch.no_grad():
    for i in range(100):
        # Slice context to block_size if needed, or pass full sequence
        input_ids = [char_to_id[c] for c in generated_text]
        text_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        
        logits = model(text_tensor) 
        last_logits = logits[0, -1, :] / temperature
        probabilities = F.softmax(last_logits, dim=-1)
        
        next_char_id = torch.multinomial(probabilities, num_samples=1).item()
        generated_text += id_to_char[next_char_id]

print("\nGenerated Text:\n", generated_text)