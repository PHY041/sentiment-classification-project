from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    num_epochs: int, 
    device: torch.device = None
) -> Tuple[List[float], List[float]]:
    """
    Train the model and evaluate on the validation set after each epoch.
    Returns:
        train_losses (List[float]): List of training losses per epoch.
        val_accuracies (List[float]): List of validation accuracies per epoch.
    """
    if device is None:
        device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        
    model.to(device)
    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Multiply loss by batch size
            epoch_loss += loss.item() * inputs.size(0)
        # Compute average loss per sample
        epoch_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        val_acc = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return train_losses, val_accuracies

def evaluate_model(
    model: nn.Module, 
    data_loader: DataLoader, 
    device: torch.device
) -> float:
    """
    Evaluate the model on the given dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

def test_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device
) -> float:
    """
    Test the model on the test set.
    """
    model.load_state_dict(torch.load('best_model.pt'))
    model.to(device)
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc
