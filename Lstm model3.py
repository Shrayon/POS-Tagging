#LSTM POS tagger model 3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import re
from conllu import parse_incr
import numpy as np
import gensim.downloader as api
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
word2vec_model = api.load("word2vec-google-news-300")
file_path = '/content/en_atis-ud-train.conllu'
word_counts = defaultdict(int)
unique_pos_tags = set()
def parse_conllu(file_path):
    with open(file_path, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            if not line.startswith('#') and line.strip():
                parts = line.split('\t')
                if len(parts) > 3:
                    word = parts[1].lower()
                    pos_tag = parts[3]
                    word_counts[word] += 1
                    unique_pos_tags.add(pos_tag)
parse_conllu(file_path)
vocabulary = {word for word, count in word_counts.items() if count >= 3}
def replace_low_freq_words(sentence):
    return ['<UNK>' if word.lower() not in vocabulary else word.lower() for word in sentence]
def prepare_sequence(seq, to_ix, is_pos=False): #false if for words ture is for pos tags
    if is_pos:
        idxs = [to_ix.get(word, to_ix['<UNK>']) for word in seq]
    else:
        idxs = [to_ix.get(word, to_ix['<UNK>']) for word in seq]
    return torch.tensor(idxs, dtype=torch.long)
def get_word_vector(word, model):
    if word in model.key_to_index:
        return model[word]
    else:
        return np.zeros(model.vector_size)
def sentences_to_vectors(padded_sentences, word_to_ix, model):
    sentence_vectors = torch.zeros(padded_sentences.size(0), padded_sentences.size(1), model.vector_size)
    for i, sentence in enumerate(padded_sentences):
        for j, word_idx in enumerate(sentence):
            word = list(word_to_ix.keys())[list(word_to_ix.values()).index(word_idx.item())] if word_idx.item() in list(word_to_ix.values()) else "<UNK>"
            word_vector = get_word_vector(word, model)
            sentence_vectors[i, j, :] = torch.tensor(word_vector)
    return sentence_vectors
data_file = open("/content/en_atis-ud-train.conllu", "r", encoding="utf-8")
sentences = []
pos_tags = []
for tokenlist in parse_incr(data_file):
    sentence_tokens = []
    sentence_pos_tags = []
    for token in tokenlist:
        sentence_tokens.append(token['form'])
        sentence_pos_tags.append(token['upos'])
    sentences.append(replace_low_freq_words(sentence_tokens))
    pos_tags.append(sentence_pos_tags)
max_sentence_length = max(len(sentence) for sentence in sentences)
pos_tag_vocab = {tag: idx for idx, tag in enumerate(unique_pos_tags)}
pos_tag_vocab['<PAD>'] = len(pos_tag_vocab)
pos_tag_vocab['<UNK>'] = len(pos_tag_vocab)
word_to_ix = {word: i for i, word in enumerate(vocabulary, start=1)}
word_to_ix['<UNK>'] = 0
word_to_ix['<PAD>'] = len(word_to_ix)
indexed_sentences = [prepare_sequence(sentence, word_to_ix) for sentence in sentences]
indexed_pos_tags = [prepare_sequence(tags, pos_tag_vocab, is_pos=True) for tags in pos_tags]
pad_token = word_to_ix['<PAD>'] #padding
pad_tag = pos_tag_vocab['<PAD>'] #padding
padded_sentences = pad_sequence(indexed_sentences, batch_first=True, padding_value=pad_token) #adding the padding to sentences
padded_pos_tags = pad_sequence(indexed_pos_tags, batch_first=True, padding_value=pad_tag) #adding the padding to the words
sentence_vectors = sentences_to_vectors(padded_sentences, word_to_ix, word2vec_model)
class LSTMPOSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMPOSTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=3, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.elu = nn.ELU()
    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2tag(self.elu(lstm_out))
        tag_scores = nn.functional.log_softmax(tag_space, dim=2)
        return tag_scores
EMBEDDING_DIM = word2vec_model.vector_size
HIDDEN_DIM = 256
VOCAB_SIZE = len(word_to_ix)
TAGSET_SIZE = len(pos_tag_vocab)
model = LSTMPOSTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAGSET_SIZE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
target_sequences = padded_pos_tags
for epoch in range(10):
    for sentence_in, targets in zip(sentence_vectors, target_sequences):
        model.zero_grad()
        tag_scores = model(sentence_in.unsqueeze(0))
        loss = loss_function(tag_scores.view(-1, TAGSET_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
def parse_conllu_test(file_path):
    sentences = []
    pos_tags = []
    with open(file_path, 'r', encoding='utf-8') as data_file:
        for tokenlist in parse_incr(data_file):
            sentence_tokens = []
            sentence_pos_tags = []
            for token in tokenlist:
                sentence_tokens.append(token['form'])
                sentence_pos_tags.append(token['upos'])
            sentences.append(replace_low_freq_words(sentence_tokens))
            pos_tags.append(sentence_pos_tags)
    return sentences, pos_tags
def predict_tags(sentence_vectors, model):
    model.eval()
    predictions = []
    for sentence in sentence_vectors:
        with torch.no_grad():
            tag_scores = model(sentence.unsqueeze(0))
            predicted_tags = torch.argmax(tag_scores, dim=2)
            predictions.append(predicted_tags.squeeze(0))
    return predictions
def compute_metrics(predictions, true_tags, pos_tag_vocab):
    all_predictions = torch.cat([pred.view(-1) for pred in predictions]).cpu().numpy()
    all_true_tags = true_tags.view(-1).cpu().numpy()
    accuracy = accuracy_score(all_true_tags, all_predictions)
    recall_micro = recall_score(all_true_tags, all_predictions, average='micro', zero_division=0)
    recall_macro = recall_score(all_true_tags, all_predictions, average='macro', zero_division=0)
    f1_micro = f1_score(all_true_tags, all_predictions, average='micro', zero_division=0)
    f1_macro = f1_score(all_true_tags, all_predictions, average='macro', zero_division=0)
    return accuracy, recall_micro, recall_macro, f1_micro, f1_macro


def generate_confusion_matrix(predictions, true_tags_tensor, pos_tag_vocab):
    all_predictions = torch.cat([pred.view(-1) for pred in predictions]).cpu().numpy()
    all_true_tags = true_tags_tensor.view(-1).cpu().numpy()
    cm = confusion_matrix(all_true_tags, all_predictions, labels=list(pos_tag_vocab.values()))
    non_pad_labels = [label for label in pos_tag_vocab if label not in ['<PAD>', '<UNK>']]
    non_pad_indices = [pos_tag_vocab[label] for label in non_pad_labels]
    filtered_cm = cm[np.ix_(non_pad_indices, non_pad_indices)]
    filtered_labels = [non_pad_labels[i] for i in range(len(non_pad_indices))]
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(confusion_matrix=filtered_cm, display_labels=filtered_labels).plot(values_format='d', ax=ax, cmap='viridis')
    plt.xticks(rotation=90)
    plt.show()
test_file_path = "/content/en_atis-ud-dev.conllu"
test_sentences, test_pos_tags = parse_conllu_test(test_file_path)
indexed_sentences_test = [prepare_sequence(sentence, word_to_ix) for sentence in test_sentences]
indexed_pos_tags_test = [prepare_sequence(tags, pos_tag_vocab, is_pos=True) for tags in test_pos_tags]
padded_sentences_test = pad_sequence(indexed_sentences_test, batch_first=True, padding_value=pad_token)
padded_pos_tags_test = pad_sequence(indexed_pos_tags_test, batch_first=True, padding_value=pad_tag)
sentence_vectors_test = sentences_to_vectors(padded_sentences_test, word_to_ix, word2vec_model)
target_sequences_test = padded_pos_tags_test
predictions_test = predict_tags(sentence_vectors_test, model)
accuracy, recall_micro, recall_macro, f1_micro, f1_macro = compute_metrics(predictions_test, target_sequences_test, pos_tag_vocab)
print(f'Accuracy: {accuracy:.4f}')
print(f'Macro Recall: {recall_macro:.4f}')
print(f'Macro F1: {f1_macro:.4f}')
print(f'Micro Recall: {recall_micro:.4f}')
print(f'Micro F1: {f1_micro:.4f}')
generate_confusion_matrix(predictions_test, target_sequences_test, pos_tag_vocab)