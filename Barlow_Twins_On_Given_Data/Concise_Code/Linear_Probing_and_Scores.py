class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)
#laballed dataset  
class LabeledDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

  #linear probing
  model.eval()
features = []
labels = []
with torch.no_grad():
    for images, lbls in tqdm(val_loader):
        images = images.to(device)
        feats = model.backbone(images)
        features.append(feats.cpu().numpy())
        labels.extend(lbls.numpy())

features = np.concatenate(features, axis=0)
labels = np.array(labels)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(features, labels)

# Predictions
preds = clf.predict(features)

# Evaluation
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average='weighted')
print(f'Linear Probing Accuracy: {acc:.4f}')
print(f'Linear Probing F1 Score: {f1:.4f}')

# Dimensionality reduction with t-SNE (2D)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_2d = tsne.fit_transform(features)

# Plot true labels
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
scatter = plt.scatter(features_2d[:,0], features_2d[:,1], c=labels, cmap='tab20', s=10)
plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE of Features Colored by True Labels')

# Plot predicted labels
plt.subplot(1,2,2)
scatter = plt.scatter(features_2d[:,0], features_2d[:,1], c=preds, cmap='tab20', s=10)
plt.legend(*scatter.legend_elements(), title="Predicted Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE of Features Colored by Predicted Labels')

plt.tight_layout()
plt.show()


      
