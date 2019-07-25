import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        
        # set random seed
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(self.input_size, 300)
        self.gru = nn.GRU(300, self.hidden_size)
    
    def forward(self, input, hidden=None):
        embedded = self.embedding(input).view(1, input.shape[0], -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden