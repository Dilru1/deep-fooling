import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class HistoryCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, n_stack=3):
        super().__init__(observation_space, features_dim)

        # 1. We receive a flattened shape, e.g., (57,)
        total_input_dim = observation_space.shape[0]
        
        # 2. We calculate the original feature size: 57 / 3 = 19
        self.n_stack = n_stack
        self.n_features = total_input_dim // n_stack
        
        #print(f"DEBUG: Input {total_input_dim} -> Reshaping to ({self.n_stack}, {self.n_features})")

        self.cnn = nn.Sequential(
            # Input: (Batch, Channels=19, Length=3)
            nn.Conv1d(in_channels=self.n_features, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output size dynamically
        with torch.no_grad():
            # Dummy input to calculate flattened size
            # We simulate the exact reshaping logic used in forward()
            dummy_flat = torch.zeros(1, total_input_dim)
            
            # Reshape: (Batch, Stack, Features) -> (Batch, Features, Stack)
            dummy_reshaped = dummy_flat.view(1, self.n_stack, self.n_features).permute(0, 2, 1)
            
            n_flatten = self.cnn(dummy_reshaped).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations):
        # 1. Input is (Batch, 57)
        # 2. Reshape to (Batch, Stack=3, Features=19)
        x = observations.view(-1, self.n_stack, self.n_features)
        
        # 3. Permute for CNN: (Batch, Features=19, Stack=3)
        # We treat Features as Channels, and Stack as Time/Length
        x = x.permute(0, 2, 1)
        
        x = self.cnn(x)
        return self.linear(x)
