import torch
import torch.nn as nn
import torch.nn.functional as F

class SDG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SDG, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output probabilities for selecting instances
        )
    
    def forward(self, x):
        x = self.network(x)
        return x

class FeatureExtractor1(nn.Module):
    def __init__(self):
        super(FeatureExtractor1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))  # [batch_size, 32, 28, 28]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 32, 14, 14]
        x = F.relu(self.conv2(x))  # [batch_size, 64, 14, 14]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 64, 7, 7]
        return x

class FeatureExtractor2(nn.Module):
    def __init__(self):
        super(FeatureExtractor2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256*3*3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input shape: [batch_size, 64, 7, 7]
        x = F.relu(self.conv1(x))  # [batch_size, 128, 7, 7]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 128, 3, 3]
        x = F.relu(self.conv2(x))  # [batch_size, 256, 3, 3]
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(256*3*3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input shape: [batch_size, 256, 3, 3]
        x = x.view(-1, 256*3*3).clone()  # Reshape to [batch_size, 256*3*3]
        x = F.relu(self.fc1(x))  # [batch_size, 128]
        x = self.fc2(x)  # [batch_size, 10] (10 classes)
        return x

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.feature_extractor1 = FeatureExtractor1()
        self.feature_extractor2 = FeatureExtractor2()
        self.classifier = Classifier()

    def forward(self, x):
        features1 = self.feature_extractor1(x)
        features2 = self.feature_extractor2(features1)
        out = self.classifier(features2)
        return features1, features2, out

# Instantiate the model
# model = CNN()
