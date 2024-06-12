
import torch
import torch.nn as nn
import torch.optim as optim

class SDG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SDG, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output probabilities for selecting instances
        )
    
    def forward(self, x):
        return self.network(x)

class FeatureExtractor1(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FeatureExtractor1, self).__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim  # Update input dimension for next layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FeatureExtractor2(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FeatureExtractor2, self).__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim  # Update input dimension for next layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.network = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.network(x))

class Predictor(nn.Module):
    def __init__(self, input_dim, feature_dims1, feature_dims2, classifier_output_dim):
        super(Predictor, self).__init__()
        self.feature_extractor1 = FeatureExtractor1(input_dim, feature_dims1)
        self.feature_extractor2 = FeatureExtractor2(feature_dims1[-1], feature_dims2)
        self.classifier = Classifier(feature_dims2[-1], classifier_output_dim)

    def forward(self, x):
        features1 = self.feature_extractor1(x)
        features2 = self.feature_extractor2(features1)
        return self.classifier(features2)

# def joint_train(sdg, predictor, data_loader, optimizer, epochs=10):
#     criterion = nn.BCELoss()
#     for epoch in range(epochs):
#         total_loss = 0
#         for data, targets in data_loader:
#             optimizer.zero_grad()

#             # Predict selection probabilities and apply them
#             selection_probs = sdg(data)
#             selected_indices = selection_probs > 0.5
#             selected_data = data[selected_indices.squeeze(1)]
#             selected_targets = targets[selected_indices.squeeze(1)]

#             if len(selected_data) > 0:
#                 predictions = predictor(selected_data)
#                 loss = criterion(predictions, selected_targets)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()

#         print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

# if __name__ == "__main__":
#     input_dim = 128
#     feature_dims = [128, 64]
#     classifier_output_dim = 1
#     predictor = Predictor(input_dim, feature_dims, classifier_output_dim)
#     sdg = SDG(input_dim, 128, 1)
#     optimizer = optim.Adam(list(sdg.parameters()) + list(predictor.parameters()), lr=0.001)

#     # Dummy data loader (simulated environment)
#     batch_size = 20
#     dummy_loader = [(torch.randn(batch_size, input_dim), torch.rand(batch_size, 1)) for _ in range(50)]

#     # Train the model
#     joint_train(sdg, predictor, dummy_loader, optimizer, epochs=5)
