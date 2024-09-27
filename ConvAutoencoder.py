import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=2),  # Layer 1: Convolution
#             nn.ReLU(),
            nn.SELU(),  # Activation function as specified to use SELU
            nn.MaxPool2d(2, stride=2),  # Layer 2: Max Pooling
            nn.Conv2d(20, 50, kernel_size=3, stride=1, padding=1),  # Layer 3: Convolution
#             nn.ReLU(),
            nn.SELU(),  # SELU activation
            nn.MaxPool2d(2, stride=2),  # Layer 4: Max Pooling
            nn.Conv2d(50, 64, kernel_size=3, stride=1, padding=1),  # Layer 5: Convolution
#             nn.ReLU(),
            nn.SELU(),  # SELU activation
            nn.MaxPool2d(2, stride=2)   # Layer 6: Max Pooling
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Layer 8: Transposed Convolution
#             nn.ReLU(),
            nn.SELU(),  # SELU activation
            nn.ConvTranspose2d(64, 50, kernel_size=3, stride=2, padding=1, output_padding=1),  # Layer 10: Transposed Convolution
#             nn.ReLU(),
            nn.SELU(),  # SELU activation
            nn.ConvTranspose2d(50, 20, kernel_size=3, stride=2, padding=1, output_padding=1),  # Layer 11: Transposed Convolution
#             nn.ReLU(),
            nn.SELU(),  # SELU activation
            nn.ConvTranspose2d(20, 3, kernel_size=5, stride=1, padding=2),  # Layer 12: Transposed Convolution
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
