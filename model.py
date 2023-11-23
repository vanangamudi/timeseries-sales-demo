import torch
from torch import nn

import logger
log = logger.get_logger(__file__, 'INFO')

class LSTMModel(nn.Module):

    def __init__(self, config, hpconfig, input_size, output_size):
        super().__init__()

        self.hpconfig = hpconfig
        
        self.input_size = input_size
        self.output_size = output_size

        self.num_layers = hpconfig['num_layers']
        self.hidden_size = hpconfig['hidden_size']

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size * self.num_layers, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        h_0, c_0 = h_0.to(x.device), c_0.to(x.device)
                
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.num_layers * self.hidden_size)
        
        out = self.fc(torch.relu(h_out))
        
        return out
