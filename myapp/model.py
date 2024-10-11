import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_classes=47):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),  # (batch_size, 1, 28, 28) -> (batch_size, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (batch_size, 1, 28, 28) -> (batch_size, 32, 14, 14)
            nn.Conv2d(32, 64, (3, 3)),  # (batch_size, 32, 14, 14) -> (batch_size, 64, 12, 12)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (batch_size, 64, 12, 12) -> (batch_size, 64, 6, 6)
            nn.Flatten(),  # (batch_size, 64, 6, 6) -> (batch_size, 64*6*6)
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.model(x)
