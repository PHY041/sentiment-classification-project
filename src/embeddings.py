# src/embeddings.py

from typing import Dict, Set
import numpy as np
import os

def load_pretrained_vocab(
    embedding_file: str = 'glove.6B.300d.txt'
) -> Set[str]:
    """
    Load the vocabulary from the GloVe embedding file.
    """
    vocab = set()
    if not os.path.isfile(embedding_file):
        raise FileNotFoundError(f"Embedding file '{embedding_file}' not found.")
    with open(embedding_file, 'r', encoding='utf8') as f:
        for line in f:
            word = line.strip().split()[0]
            vocab.add(word)
    return vocab

def load_pretrained_embeddings(
    vocab: Dict[str, int], 
    embedding_dim: int = 300, 
    embedding_file: str = 'glove.6B.300d.txt'
) -> np.ndarray:
    """
    Load GloVe embeddings and create an embedding matrix aligned with the vocabulary indices.
    """
    embeddings_index = {}
    if not os.path.isfile(embedding_file):
        raise FileNotFoundError(f"Embedding file '{embedding_file}' not found.")
    with open(embedding_file, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    # Prepare embedding matrix
    embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype='float32')
    for word, idx in vocab.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            # OOV words get random vectors
            embedding_matrix[idx] = np.random.uniform(-0.05, 0.05, embedding_dim)
    return embedding_matrix
