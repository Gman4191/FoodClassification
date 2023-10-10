import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

class DataPreprocessor:
    def __init__(self, dataset_path):
        
        self.path = dataset_path

    def transform_data(self, batch_size = 64):
        transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(self.path, transform)

        # Define the sizes of your training, validation, and testing sets
        train_size = int(0.6 * len(dataset))  # 60% for training
        val_size = int(0.2 * len(dataset))    # 20% for validation
        test_size = len(dataset) - train_size - val_size  # 20% for evaluation

        # Use random_split to split the dataset into training, validation, and testing sets
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Create DataLoader for each set
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader