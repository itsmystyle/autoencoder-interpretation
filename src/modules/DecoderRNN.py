import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size=300, use_same_embedding=False):
        super(DecoderRNN, self).__init__()
        
        # set random seed
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.use_same_embedding = use_same_embedding
        
        if not self.use_same_embedding:
            self.embedding = nn.Embedding(self.output_size, self.embedding_size)
            
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        if not self.use_same_embedding:
            output = self.embedding(input).view(1, input.shape[0], -1)
        else:
            output = input
            
#         output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        
        return output, hidden