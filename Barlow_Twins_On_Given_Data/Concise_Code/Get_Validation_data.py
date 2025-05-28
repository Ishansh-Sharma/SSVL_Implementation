import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# selected_classes = list(labels_map.keys())[:40]

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

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_root_dir = '/kaggle/input/ssl-dataset/ssl_dataset/valid'

# Build validation dataset only with selected classes
val_dataset = LabeledDataset(valid_root_dir, selected_classes, transform=val_transform)
print(f'Validation dataset size: {len(val_dataset)}')

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
