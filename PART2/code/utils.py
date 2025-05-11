import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import random
import seaborn as sns

def get_random_test_examples(test_dataset, n_normal=5, n_pneumonia=5, seed=None):
    """
    Randomly select indices of normal and pneumonia samples from the test dataset.
    
    Args:
        test_dataset: The test dataset
        n_normal: Number of normal samples to select
        n_pneumonia: Number of pneumonia samples to select
        seed: Random seed for reproducibility (optional)
    
    Returns:
        normal_indices: List of indices for normal samples
        pneumonia_indices: List of indices for pneumonia samples
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        
    # Collect all indices by class
    all_normal_indices = []
    all_pneumonia_indices = []
    
    # Scan the dataset to find indices for each class
    for i, (_, label) in enumerate(test_dataset):
        if label == 0:  # Normal class
            all_normal_indices.append(i)
        elif label == 1:  # Pneumonia class
            all_pneumonia_indices.append(i)
    
    # Print available samples
    print(f"Found {len(all_normal_indices)} normal samples and {len(all_pneumonia_indices)} pneumonia samples")
    
    # Check if we have enough samples
    if len(all_normal_indices) < n_normal:
        print(f"Warning: Only {len(all_normal_indices)} normal samples available, less than requested {n_normal}")
        n_normal = len(all_normal_indices)
        
    if len(all_pneumonia_indices) < n_pneumonia:
        print(f"Warning: Only {len(all_pneumonia_indices)} pneumonia samples available, less than requested {n_pneumonia}")
        n_pneumonia = len(all_pneumonia_indices)
    
    # Randomly select indices
    normal_indices = random.sample(all_normal_indices, n_normal)
    pneumonia_indices = random.sample(all_pneumonia_indices, n_pneumonia)
    
    # Print selected indices
    print(f"Selected normal indices: {normal_indices}")
    print(f"Selected pneumonia indices: {pneumonia_indices}")
    
    return normal_indices, pneumonia_indices

# Function to get examples of each class from test set
def get_test_examples(test_dataset, n_normal=5, n_pneumonia=5):
    normal_indices = []
    pneumonia_indices = []
    
    for i, (_, label) in enumerate(test_dataset):
        if label == 0 and len(normal_indices) < n_normal:  # Normal class
            normal_indices.append(i)
        elif label == 1 and len(pneumonia_indices) < n_pneumonia:  # Pneumonia class
            pneumonia_indices.append(i)
        
        if len(normal_indices) >= n_normal and len(pneumonia_indices) >= n_pneumonia:
            break
    
    return normal_indices, pneumonia_indices


# Define transforms
def transform_imgs():

    # Define transforms (same as used during training)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),      # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Dataset for visualization (without normalization)
    viz_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return train_transform, test_transform, viz_transform


# Check class distribution    
def count_class_dist(dataset):
    train_counts = [0, 0]
    for _, label in dataset:
        train_counts[label] += 1

    print(f"Training set - Normal: {train_counts[0]}, Pneumonia: {train_counts[1]}")
    print(f"Class weights - Normal: {len(dataset)/train_counts[0]}, Pneumonia: {len(dataset)/train_counts[1]}")
    return train_counts

    
# Training model
def train_model(model, train_dataset, train_loader, val_dataset, val_loader, criterion, optimizer, scheduler, model_name, device, num_epochs=10):
    start_time = time.time()
    best_val_loss = float('inf')
    
    # To store training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        # No gradients needed for validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_dataset)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())
        
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # Update learning rate
        scheduler.step(val_epoch_loss)
        
        # Save best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), 'cnn/best_' + model_name + '.pth')
            print("Saved best model!")
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Load best model
    model.load_state_dict(torch.load('cnn/best_' + model_name + '.pth'))
    return model

    
# Evaluate model on test set
def evaluate_model(model, test_loader, device, model_name=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Print results
    print("\nTest Set Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if model_name != None:
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Pneumonia'],
                    yticklabels=['Normal', 'Pneumonia'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('figs/confusion_matrix_' + model_name + '.png')
        plt.show()
    
    return accuracy, precision, recall, f1, cm