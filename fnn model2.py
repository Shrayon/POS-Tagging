#FNN MODEL 2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from io import open
import numpy as np
import gensim.downloader as api
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from io import open
import gensim.downloader as api
def preprocess_conllu_file_test(file_path, p, s, low_freq_words, seen_pos_tags):
    input_sequences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as data_file:
        sentence_tokens = []
        skip_sentence = False
        for line in data_file:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('\t')
                if len(parts) > 3:
                    word = parts[1].lower()
                    pos_tag = parts[3]
                    
                    if pos_tag not in seen_pos_tags:
                        skip_sentence = True
                        sentence_tokens = []  
                        continue
                    word = word if word not in low_freq_words else '<unknown>'
                    sentence_tokens.append((word, pos_tag))
            elif sentence_tokens and not skip_sentence:
                words = ['<start>'] + ['<PAD>'] * (p-1) + [token[0] for token in sentence_tokens] + ['<PAD>'] * (s-1) + ['<end>']
                pos_tags = ['<PAD>'] * (p-1) + [token[1] for token in sentence_tokens] + ['<PAD>'] * (s-1)
                for i in range(len(sentence_tokens)):
                    segment_start = i
                    segment_end = i + p + s
                    segment = words[segment_start:segment_end+1]
                    label = pos_tags[i + p - 1]
                    input_sequences.append(segment)
                    labels.append(label)
                sentence_tokens = []
                skip_sentence = False  
    return input_sequences, labels
def parse_conllu(file_path):
    word_counts = defaultdict(int)
    unique_pos_tags = set()
    with open(file_path, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            if not line.startswith('#') and line.strip():
                parts = line.split('\t')
                if len(parts) > 3:
                    word = parts[1].lower()
                    pos_tag = parts[3]
                    word_counts[word] += 1
                    unique_pos_tags.add(pos_tag)
    return word_counts, unique_pos_tags
def preprocess_conllu_file(file_path, p, s, low_freq_words):
    input_sequences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as data_file:
        sentence_tokens = []
        for line in data_file:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('\t')
                if len(parts) > 3:
                    word = parts[1].lower()
                    pos_tag = parts[3]
                    word = word if word not in low_freq_words else '<unknown>'
                    sentence_tokens.append((word, pos_tag))
            elif sentence_tokens:
                words = ['<start>'] + ['<PAD>'] * (p-1) + [token[0] for token in sentence_tokens] + ['<PAD>'] * (s-1) + ['<end>']
                pos_tags = ['<PAD>'] * (p-1) + [token[1] for token in sentence_tokens] + ['<PAD>'] * (s-1)
                for i in range(len(sentence_tokens)):
                    segment_start = i
                    segment_end = i + p + s
                    segment = words[segment_start:segment_end+1]
                    label = pos_tags[i + p - 1]
                    input_sequences.append(segment)
                    labels.append(label)
                sentence_tokens = []
    return input_sequences, labels
def pos_tag_to_one_hot(tag, pos_tag_to_index):
    one_hot_vector = np.zeros(len(pos_tag_to_index))
    one_hot_vector[pos_tag_to_index[tag]] = 1
    return one_hot_vector
def get_word_embedding(word, word2vec_model):
    if word in word2vec_model.key_to_index:
        return word2vec_model[word]
    else:
        return np.zeros(word2vec_model.vector_size)
def preprocess_with_embeddings_and_one_hot(input_sequences, labels, model, pos_tag_to_index):
    embeddings_sequences = []
    one_hot_labels = []
    for sequence in input_sequences:
        embeddings_sequence = [get_word_embedding(word, model) for word in sequence]
        embeddings_sequences.append(embeddings_sequence)
    for label in labels:
        one_hot_label = pos_tag_to_one_hot(label, pos_tag_to_index)
        one_hot_labels.append(one_hot_label)
    return np.array(embeddings_sequences), np.array(one_hot_labels)
class POSFFNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(POSFFNN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * (p + s + 1), hidden_dim )
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim , hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
file_path = '/content/en_atis-ud-train.conllu'
word2vec_model= api.load("word2vec-google-news-300")
p = 2
s = 2
word_counts, unique_pos_tags = parse_conllu(file_path)
low_frequency_words = {word for word, count in word_counts.items() if count < 3}
input_sequences, labels = preprocess_conllu_file(file_path, p, s, low_frequency_words)
pos_tag_to_index = {tag: idx for idx, tag in enumerate(sorted(unique_pos_tags))}
input_sequences, labels = preprocess_with_embeddings_and_one_hot(input_sequences, labels, word2vec_model, pos_tag_to_index)
input_sequences_flattened = input_sequences.reshape(input_sequences.shape[0], -1)
labels_indices = np.argmax(labels, axis=1)
X_tensor = torch.tensor(input_sequences_flattened, dtype=torch.float)
y_tensor = torch.tensor(labels_indices, dtype=torch.long)
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
embedding_dim = word2vec_model.vector_size
hidden_dim =  512
output_dim = len(pos_tag_to_index)
model = POSFFNN(embedding_dim, hidden_dim, output_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')
test_file_path = '/content/en_atis-ud-dev.conllu'
word_counts_test, unique_pos_tags_test = parse_conllu(test_file_path)
seen_pos_tags = unique_pos_tags
input_sequences_test, labels_test = preprocess_conllu_file_test(test_file_path, p, s, low_frequency_words, seen_pos_tags)
input_sequences_test, labels_test = preprocess_with_embeddings_and_one_hot(input_sequences_test, labels_test, word2vec_model, pos_tag_to_index)
input_sequences_test_flattened = input_sequences_test.reshape(input_sequences_test.shape[0], -1)
labels_indices_test = np.argmax(labels_test, axis=1)
X_tensor_test = torch.tensor(input_sequences_test_flattened, dtype=torch.float)
y_tensor_test = torch.tensor(labels_indices_test, dtype=torch.long)
dataset_test = torch.utils.data.TensorDataset(X_tensor_test, y_tensor_test)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False)
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in dataloader_test:
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)
        predictions.extend(predicted_labels.tolist())
        true_labels.extend(labels.tolist())
predictions = np.array(predictions)
true_labels = np.array(true_labels)
accuracy = accuracy_score(true_labels, predictions)
recall_macro = recall_score(true_labels, predictions, average='macro')
f1_macro = f1_score(true_labels, predictions, average='macro')
recall_micro = recall_score(true_labels, predictions, average='micro')
f1_micro = f1_score(true_labels, predictions, average='micro')
print(f'Accuracy: {accuracy:.4f}')
print(f'Macro Recall: {recall_macro:.4f}')
print(f'Macro F1: {f1_macro:.4f}')
print(f'Micro Recall: {recall_micro:.4f}')
print(f'Micro F1: {f1_micro:.4f}')
conf_matrix = confusion_matrix(true_labels, predictions)
sorted_pos_tags = sorted(pos_tag_to_index, key=pos_tag_to_index.get)
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted_pos_tags, yticklabels=sorted_pos_tags)
plt.title('Confusion Matrix')
plt.ylabel('Labels')
plt.xlabel('Predicted Labels')
plt.show()