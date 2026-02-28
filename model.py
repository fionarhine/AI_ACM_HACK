#hello this is the model file (for making our model more accurate)

# ==========================================
# 4. MODEL DEFINITION & HYPERPARAMETERS
# ==========================================

class StarWarsClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(StarWarsClassifier, self).__init__()
        
        # Load a pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=True)

        # Optional: Freeze the early layers so they don't update during training.
        # This speeds up training and prevents the model from forgetting basic features.
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Replace the final fully connected layer of the ResNet18 model
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
    def forward(self, x):
        return self.model(x)

# Initialize Model, Loss Function, and Optimizer
model = StarWarsClassifier(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()

# --- PLAY WITH THESE HYPERPARAMETERS ---
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 5
# ---------------------------------------

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. THE TRAINING LOOP
# ==========================================

# Split data 80/20 for training and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

def train_one_epoch(model, dataloader):
    model.train()
    total_loss = 0

    for batch in dataloader:
        imgs, labels, _ = batch
        imgs, labels = imgs.to(device), labels.to(device)

        # TODO: Implement the 5 crucial steps of PyTorch training!
        # 1. Clear the old gradients
        # 2. Forward pass: generate predictions
        # 3. Calculate the loss using the criterion
        # 4. Backward pass: compute gradients
        # 5. Update the model weights using the optimizer
        
        # total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            imgs, labels, _ = batch
            imgs, labels = imgs.to(device), labels.to(device)
            
            # TODO: Generate outputs, get the predicted classes, and calculate accuracy
            
    return 0.0 # Return correct/total when implemented

# Start Training!
print("Starting Training...")
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss = {train_loss:.4f} | Validation Acc = {val_acc:.2%}")


