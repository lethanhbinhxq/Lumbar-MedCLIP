import pandas as pd
import numpy as np
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT, MedCLIPVisionModelSwin
from medclip import MedCLIPProcessor
from torch.utils.data import WeightedRandomSampler

# Load MedCLIP model and processor
model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model.from_pretrained('./checkpoints/vision_text_pretrain/best')
model.cuda()
model.eval()
processor = MedCLIPProcessor()

# # Custom dataset
# class MRIDataset(Dataset):
#     def __init__(self, csv_path):
#         self.data = pd.read_csv(csv_path)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         image = Image.open(row['imgpath']).convert("RGB")
#         text = row['report']
#         label = torch.tensor([row['No Finding'], row['LBP']], dtype=torch.float32)
#         inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
#         with torch.no_grad():
#             img_emb = model.encode_image(inputs['pixel_values'].cuda())
#         return img_emb.squeeze(0), label
        
#     def get_weighted_sampler(self):
#         """Create a WeightedRandomSampler based on the class distribution."""
#         labels = self.data[['No Finding', 'LBP']].values  # shape [N, 2]
#         single_labels = np.argmax(labels, axis=1)  # convert to single label (0 or 1)

#         class_sample_count = np.bincount(single_labels)
#         class_sample_count[class_sample_count == 0] = 1  # avoid divide-by-zero

#         weight_per_class = 1. / class_sample_count
#         weights = weight_per_class[single_labels]

#         sampler = WeightedRandomSampler(
#             weights=weights,
#             num_samples=len(weights),
#             replacement=True
#         )
#         return sampler

class RawMRIDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['imgpath']).convert("RGB")
        text = row['report']
        label = torch.tensor([row['No Finding'], row['LBP']], dtype=torch.float32)
        return image, text, label

def collate_fn(batch):
    images, texts, labels = zip(*batch)  # unzip list of tuples
    return list(images), list(texts), torch.stack(labels)

def compute_embeddings(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_embeddings = []
    all_labels = []

    for images, texts, labels in loader:
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            img_embs = model.encode_image(inputs['pixel_values'].cuda(), use_projection = False)
        all_embeddings.append(img_embs.cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels

class InMemoryDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# # Dataloaders
# train_dataset = MRIDataset('./local_data/lbp-train-meta.csv')
# valid_dataset = MRIDataset('./local_data/lbp-valid-meta.csv')
# test_dataset  = MRIDataset('./local_data/lbp-test-meta.csv')

# train_sampler = train_dataset.get_weighted_sampler()
# valid_sampler = valid_dataset.get_weighted_sampler()

# train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=False, sampler=train_sampler)
# valid_loader = DataLoader(valid_dataset, batch_size=4096, shuffle=False, sampler=valid_sampler)
# test_loader  = DataLoader(test_dataset, batch_size=4096)

# Load raw datasets
train_raw = RawMRIDataset('./local_data/lbp-train-meta.csv')
valid_raw = RawMRIDataset('./local_data/lbp-valid-meta.csv')
test_raw  = RawMRIDataset('./local_data/lbp-test-meta.csv')

# Precompute embeddings once in memory
train_emb, train_lbl = compute_embeddings(train_raw)
valid_emb, valid_lbl = compute_embeddings(valid_raw)
test_emb, test_lbl   = compute_embeddings(test_raw)

# Create final datasets
train_dataset = InMemoryDataset(train_emb, train_lbl)
valid_dataset = InMemoryDataset(valid_emb, valid_lbl)
test_dataset  = InMemoryDataset(test_emb, test_lbl)

# Sampler and loader
def get_sampler(labels):
    binary_labels = torch.argmax(labels, dim=1).numpy()
    class_counts = np.bincount(binary_labels)
    weights = 1.0 / class_counts[binary_labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_sampler = get_sampler(train_lbl)
valid_sampler = get_sampler(valid_lbl)

# train_loader = DataLoader(train_dataset, batch_size=4096, sampler=train_sampler)
# valid_loader = DataLoader(valid_dataset, batch_size=4096, sampler=valid_sampler)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4096, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=4096)

# MLP Classifier
# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 2)  # 2 classes: No Finding, LBP
#         )

#     def forward(self, x):
#         return self.fc(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 2)
        )

    def forward(self, x):
        return self.fc(x)

# Instantiate model
embedding_dim = train_dataset[0][0].shape[0]
clf = MLPClassifier(input_dim=embedding_dim).cuda()
print(embedding_dim)

# Loss & optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(clf.parameters(), lr=1e-4)

wandb.init(
    project="lbp_medclip_mlp", 
    name="mlp_mendeley_vit_aug_projection_512",
    config={
        "epochs": 50,
        "batch_size": 4096
    }
)

# Training loop
for epoch in range(50):
    clf.train()
    train_loss = 0
    for emb, label in train_loader:
        emb, label = emb.cuda(), label.cuda()
        optimizer.zero_grad()
        outputs = clf(emb)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    # Validation
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for emb, label in valid_loader:
            emb, label = emb.cuda(), label.cuda()
            outputs = clf(emb)
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(label, dim=1)
            correct += (preds == labels).sum().item()
            total += label.size(0)
    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

    # log to wandb
    wandb.log({"train_loss": avg_train_loss, "val_accuracy": val_acc})

wandb.finish()

# torch.save(clf.state_dict(), './mlp_classifier.pth')
torch.save(clf.state_dict(), './mlp_classifier.pth')
print("✅ Saved MLP classifier to ./mlp_classifier.pth")

# Testing — get final performance + predictions
clf.eval()
pred_classes = []
true_classes = []
indices = []
idx = 0

with torch.no_grad():
    for emb, label in test_loader:
        batch_size = emb.size(0)
        emb, label = emb.cuda(), label.cuda()
        outputs = clf(emb)
        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(label, dim=1)
        
        pred_classes.extend(preds.cpu().tolist())
        true_classes.extend(labels.cpu().tolist())
        
        indices.extend(range(idx, idx + batch_size))
        idx += batch_size

from sklearn.metrics import classification_report
print(classification_report(true_classes, pred_classes, target_names=['No Finding', 'LBP']))

test_results = pd.DataFrame({
    'index': indices,
    'predicted_class': pred_classes,
    'true_class': true_classes
})
test_results.to_csv('./lbp_test_predictions.csv', index=False)
print("✅ Test predictions saved to ./lbp_test_predictions.csv")