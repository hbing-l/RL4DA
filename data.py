import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class RotNISTDataLoader:
    def __init__(self, root_path, source_angle, intermediate_angles, target_angle, shuffle_or_not, type='train', batch_size=64):
        self.root_path = root_path
        self.source_angle = source_angle
        self.intermediate_angles = intermediate_angles
        self.target_angle = target_angle
        self.batch_size = batch_size
        self.source_loader = None
        self.intermediate_loaders = []
        self.target_loader = None
        self.shuffle = shuffle_or_not
        self.type = type

    def load_data_for_angle(self, angle):
        data_path = os.path.join(self.root_path, str(angle), f'{self.type}_data_{angle}.npy')
        label_path = os.path.join(self.root_path, str(angle), f'{self.type}_label_{angle}.npy')
        domain_data = np.load(data_path)
        domain_labels = np.load(label_path)
        
        # Convert to tensors
        data_tensor = torch.tensor(domain_data, dtype=torch.float32)
        labels_tensor = torch.tensor(domain_labels, dtype=torch.long)
        
        return data_tensor, labels_tensor

    def create_dataloader(self, data, labels):
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def load_all_data(self):
        # Load the source domain data
        source_data, source_labels = self.load_data_for_angle(self.source_angle)
        # self.source_loader = self.create_dataloader(source_data, source_labels)
        self.source_loader = source_data[:100, :, :, :], source_labels[:100, :]
        # self.source_loader = source_data, source_labels

        # Load the target domain data
        target_data, target_labels = self.load_data_for_angle(self.target_angle)
        # self.target_loader = self.create_dataloader(target_data, target_labels)
        self.target_loader = target_data[:100, :, :, :], target_labels[:100, :]
        # self.target_loader = target_data, target_labels

        # Load intermediate domain data separately and store in a list
        for angle in self.intermediate_angles:
            data, labels = self.load_data_for_angle(angle)
            # loader = self.create_dataloader(data, labels)
            loader = data[:100, :, :, :], labels[:100, :]
            # loader = data, labels
            self.intermediate_loaders.append(loader)

    def get_loaders(self):
        if not self.source_loader or not self.target_loader or not self.intermediate_loaders:
            self.load_all_data()
        return self.source_loader, self.intermediate_loaders, self.target_loader


# RotNISTLoader = RotNISTDataLoader('/Users/liuhanbing/Desktop/code/RotNIST/data/angle', 0, [18, 36, 54, 72], 90, True, 'train', 256)
# source_loader, intermediate_loaders, target_loader = RotNISTLoader.get_loaders()