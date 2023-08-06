import torch.nn as nn
from functions import ReverseLayerF
from efficientnet_pytorch import EfficientNet


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0')
        num_features = self.feature_extractor._fc.in_features
        self.feature_extractor._fc = nn.Identity()  # Remove the last fully connected layer

        self.class_classifier = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 45),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha):
        feature = self.feature_extractor(input_data)
        feature = feature.view(input_data.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output