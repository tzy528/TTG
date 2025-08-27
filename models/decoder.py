import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        hidden_dim2 = hidden_dim*2
        self.num_layers = num_layers

        # Define the layers of the DNN
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)  # Add more layers if needed
        self.fc_out = nn.Linear(hidden_dim2, output_dim)
        # self.fc = nn.Linear(input_dim,output_dim)

    def forward(self, x):
        # First fully connected layer followed by ReLU activation
        x = F.tanh(self.fc1(x))
        # tanh = selu > elu
        # # Additional hidden layers
        x = F.tanh(self.fc2(x))  # Add more layers with relu if num_layers > 2
        # # Example for more layers: x = F.relu(self.fc3(x))
        #
        # # Output layer (no activation function here)
        x = self.fc_out(x)

        # x = self.fc(x)

        return x

    def MAEloss(self,origain,deco):
        loss_fn=nn.L1Loss()
        loss=loss_fn(origain,deco)
        return loss
