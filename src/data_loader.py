from typing import List, Dict, Tuple
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

nltk.download('punkt')

def preprocess_text(text: str) -> List[str]:
    """
    Tokenize and preprocess the text.
    """
    # Lowercase and tokenize
    tokens = word_tokenize(text.lower())
    # Remove non-alphabetic tokens (optional)
    tokens = [token for token in tokens if token.isalpha()]
    return tokens

def build_vocab(
    texts: List[str], 
    pre_trained_vocab: set, 
    min_freq: int = 1
) -> Dict[str, int]:
    """
    Build a vocabulary from the texts, including only words present in pre-trained vocab.
    """
    counter = Counter()
    for text in texts:
        tokens = preprocess_text(text)
        counter.update(tokens)
    # Start with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    index = len(vocab)
    for token, freq in counter.items():
        if freq >= min_freq and token in pre_trained_vocab:
            vocab[token] = index
            index += 1
    return vocab

class TextDataset(Dataset):
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        vocab: Dict[str, int], 
        max_len: int = 100
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = preprocess_text(self.texts[idx])
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def load_data(
    batch_size: int = 32, 
    max_len: int = 100
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Load data and return data loaders and vocabulary.
    """
    # Load datasets
    dataset = load_dataset("rotten_tomatoes")
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    val_texts = dataset['validation']['text']
    val_labels = dataset['validation']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    # Load pre-trained embeddings to get the vocabulary
    from src.embeddings import load_pretrained_vocab
    pre_trained_vocab = load_pretrained_vocab()

    # Build vocabulary from training data
    vocab = build_vocab(train_texts, pre_trained_vocab)

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_len)
    val_dataset = TextDataset(val_texts, val_labels, vocab, max_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, vocab
