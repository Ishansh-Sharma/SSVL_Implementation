import json

with open('/kaggle/input/ssl-dataset/ssl_dataset/Labels.json', 'r') as f:
    labels_data = json.load(f)
print("Top-level keys:", labels_data.keys())
#mapping the json file 
with open('/kaggle/input/ssl-dataset/ssl_dataset/Labels.json', 'r') as f:
    labels_map = json.load(f)
selected_classes = list(labels_map.keys())[:40]  # Select first 40 classes

#here i have described the data loader

class SSLDataset(Dataset):
    def __init__(self, root_dirs, classes, transform=None):
        self.image_paths = []
        for root_dir in root_dirs:
            for cls in classes:
                cls_dir = os.path.join(root_dir, cls)
                if os.path.isdir(cls_dir):
                    for img_name in os.listdir(cls_dir):
                        img_path = os.path.join(cls_dir, img_name)
                        self.image_paths.append(img_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            img1 = self.transform(image)
            img2 = self.transform(image)
        return img1, img2
      
#data transform

ssl_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#make trainig data
train_dirs = [
    '/kaggle/input/ssl-dataset/ssl_dataset/train.X1',
    '/kaggle/input/ssl-dataset/ssl_dataset/train.X2',
    '/kaggle/input/ssl-dataset/ssl_dataset/train.X3',
    '/kaggle/input/ssl-dataset/ssl_dataset/train.X4'
]

train_dataset = SSLDataset(train_dirs, selected_classes, transform=ssl_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
