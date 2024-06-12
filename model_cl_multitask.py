import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# class SDG(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(SDG, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#             nn.Sigmoid()  # Output probabilities for selecting instances
#         )
    
#     def forward(self, x):
#         x = self.network(x)
#         return x

class SDG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SDG, self).__init__()
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 64)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output probabilities for selecting instances
        )
    
    def forward(self, x):
        
        x = x.view(-1, 128*3*3).clone()  # Reshape to [batch_size, 128*3*3]

        x = F.relu(self.fc1(x))  # [batch_size, 256]
        x = F.relu(self.fc2(x)) # [batchsize, 64]
        
        x = self.network(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))  # [batch_size, 32, 28, 28]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 32, 14, 14]
        x = F.relu(self.conv2(x))  # [batch_size, 64, 14, 14]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 64, 7, 7]
        x1 = F.relu(self.conv3(x))  # [batch_size, 128, 7, 7]
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)  # [batch_size, 128, 3, 3]

        x2 = F.relu(self.conv4(x))  # [batch_size, 128, 7, 7]
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2)  # [batch_size, 128, 3, 3]
        return x1, x2

class FeatureExtractor1(nn.Module):
    def __init__(self):
        super(FeatureExtractor1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 64)

        self.fc3 = nn.Linear(128*3*3, 256)
        self.fc4 = nn.Linear(256, 64)

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))  # [batch_size, 32, 28, 28]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 32, 14, 14]
        x = F.relu(self.conv2(x))  # [batch_size, 64, 14, 14]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 64, 7, 7]
        x = F.relu(self.conv3(x))  # [batch_size, 128, 7, 7]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 128, 3, 3]

        x = x.view(-1, 128*3*3).clone()  # Reshape to [batch_size, 128*3*3]
        
        x1 = F.relu(self.fc1(x))  # [batch_size, 256]
        x1 = F.relu(self.fc2(x1)) # [batchsize, 64]

        x2 = F.relu(self.fc3(x))  # [batch_size, 256]
        x2 = F.relu(self.fc4(x2)) # [batchsize, 64]

        return x1, x2

class FeatureExtractor2(nn.Module):
    def __init__(self):
        super(FeatureExtractor2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 64)

        self.fc3 = nn.Linear(128*3*3, 256)
        self.fc4 = nn.Linear(256, 64)

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))  # [batch_size, 32, 28, 28]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 32, 14, 14]
        x = F.relu(self.conv2(x))  # [batch_size, 64, 14, 14]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 64, 7, 7]
        x = F.relu(self.conv3(x))  # [batch_size, 128, 7, 7]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 128, 3, 3]

        x = x.view(-1, 128*3*3).clone()  # Reshape to [batch_size, 128*3*3]

        x2 = F.relu(self.fc3(x))  # [batch_size, 256]
        x2 = F.relu(self.fc4(x2)) # [batchsize, 64]

        return x, x2


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(128*3*3, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Input shape: [batch_size, 128, 3, 3]
        x = x.view(-1, 128*3*3).clone()  # Reshape to [batch_size, 128*3*3]
        x = F.relu(self.fc1(x))  # [batch_size, 64]
        x = self.fc2(x)  # [batch_size, 10] (10 classes)
        return x
    
class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # Input shape: [batch_size, 64]
        x = F.relu(self.fc1(x))  # [batch_size, 64]
        x = self.fc2(x)  # [batch_size, 10] (10 classes)
        return x

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.feature_extractor = FeatureExtractor2()
        self.classifier = Classifier1()

    def forward(self, x):
        features1, features2 = self.feature_extractor(x)
        out = self.classifier(features2)
        return features1, features2, out

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        
        self.fc1 = nn.Linear(256*14*14, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.features(x)
        x1 = torch.flatten(x, 1)
        
        x2 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc2(x2))
        
        out = F.relu(self.fc3(x2))
        out = self.fc4(out)
        
        return x


# Instantiate the model
# model = CNN()
