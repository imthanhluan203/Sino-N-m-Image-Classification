import os
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
from transformers import AutoImageProcessor, ViTForImageClassification, AdamW, get_scheduler

# Initialize image processor and model
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=3,
    ignore_mismatched_sizes=True
)

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to tensor
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# Load dataset
dataset = ImageFolder(root="./VietNameseAncient", transform=transform)

# Shuffle dataset
torch.manual_seed(42)
indices = torch.randperm(len(dataset))
shuffled_dataset = Subset(dataset, indices)

# Split dataset into train, dev, and test sets
train_ratio = 0.8
dev_ratio = 0.1
test_ratio = 0.1

total_size = len(dataset)
train_size = int(train_ratio * total_size)
dev_size = int(dev_ratio * total_size)
test_size = total_size - train_size - dev_size

train_dataset, dev_dataset, test_dataset = random_split(shuffled_dataset, [train_size, dev_size, test_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Setup training parameters
output_dir = "./models"
os.makedirs(output_dir, exist_ok=True)
epochs = 15
gradient_accumulation_steps = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-6, weight_decay=0.01)
num_training_steps = epochs * len(train_loader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Initialize variables to track the best model
best_val_f1 = 0.0
best_model_path = os.path.join(output_dir, "best_model")

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
    
    for step, batch in enumerate(progress_bar):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(pixel_values=inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({"Loss": loss.item()})
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}")
    
    # Validation step
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch in dev_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            val_preds.extend(predictions.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    val_accuracy = accuracy_score(val_true, val_preds)
    val_f1 = f1_score(val_true, val_preds, average='weighted')
    print(f"Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")
    
    # Check if this is the best model so far
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained(best_model_path)
        print(f"Best model updated and saved to {best_model_path}")
    
    # Optionally, implement early stopping based on validation performance

# Load the best model for testing
best_model = ViTForImageClassification.from_pretrained(best_model_path)
best_model.to(device)
best_model.eval()

# Test evaluation
test_preds = []
test_true = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = best_model(pixel_values=inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        test_preds.extend(predictions.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_true, test_preds)
test_f1 = f1_score(test_true, test_preds, average='weighted')

print(f"\nTest Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
