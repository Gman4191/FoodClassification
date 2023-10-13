import unittest
import tempfile
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ..data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.dataset_path = os.path.join(self.test_dir.name, "dataset")
        os.makedirs(self.dataset_path, exist_ok=True)

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    def test_transform_data(self):
        # Create a DataPreprocessor instance
        data_preprocessor = DataPreprocessor(self.dataset_path)

        # Generate a random test dataset for the temporary directory
        test_data = torch.randn(100, 3, 256, 256)  # Assuming 100 samples

        # Save the test data to the temporary directory
        for i in range(len(test_data)):
            sample = test_data[i]
            image = transforms.ToPILImage()(sample)
            image.save(os.path.join(self.dataset_path, f"{i}.png"))

        # Call the transform_data method to get data loaders
        train_loader, val_loader, test_loader = data_preprocessor.transform_data(batch_size=64)

        # Check if the loaders are of the correct types
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

        # Check if the batch size is as expected
        self.assertEqual(train_loader.batch_size, 64)
        self.assertEqual(val_loader.batch_size, 64)
        self.assertEqual(test_loader.batch_size, 64)

        # Check if the number of samples in the loaders is as expected
        self.assertEqual(len(train_loader.dataset), int(0.6 * len(test_data)))
        self.assertEqual(len(val_loader.dataset), int(0.2 * len(test_data)))
        self.assertEqual(len(test_loader.dataset), len(test_data))

if __name__ == '__main__':
    unittest.main()
