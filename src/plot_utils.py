# Tune hyperparameter here

embedding_size = 16
use_same_embedding = True
hidden_size = 16
wordset_path = '../data/word_set_nonsence_0.pkl'
vocab_path = '../data/vocab_nonsence_0.pkl'
saved_model_path = '../models/nonsence_64_word_0/model.pkl'

use_teacher_forcing = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm

from random import shuffle
import collections
import pdb
import math

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load vocab
with open(vocab_path, 'rb') as fin:
    word_dict = pickle.load(fin)
vocab_size = len(word_dict)
idx2word = {v:k for k,v in word_dict.items()}

with open(wordset_path, 'rb') as fin:
    wordset = pickle.load(fin)
        
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
            
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        
        return output, hidden
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size=300):
        super(EncoderRNN, self).__init__()
        
        # set random seed
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
    
    def forward(self, input, hidden=None):
        output = self.embedding(input).view(1, input.shape[0], -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def get_embedding(self, input):
        return self.embedding(input).view(1, input.shape[0], -1)
    
class MyRNNCellBase(nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias, num_chunks, _wih=None, _whh=None, _bih=None, _bhh=None):
        super(MyRNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        if _wih is None:
            self.weight_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
            self.weight_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
            if bias:
                self.bias_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
                self.bias_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
            else:
                self.register_parameter('bias_ih', None)
                self.register_parameter('bias_hh', None)
            self.reset_parameters()
        else:
            self.weight_ih = nn.Parameter(torch.from_numpy(_wih))
            self.weight_hh = nn.Parameter(torch.from_numpy(_whh))
            self.bias_ih = nn.Parameter(torch.from_numpy(_bih))
            self.bias_hh = nn.Parameter(torch.from_numpy(_bhh))

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
            
class MyGRUCell(MyRNNCellBase):
    
    def __init__(self, input_size, hidden_size, bias=True, _wih=None, _whh=None, _bih=None, _bhh=None):
        if _wih is None:
            super(MyGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)
        else:
            super(MyGRUCell, self).__init__(input_size, hidden_size, bias, 3, _wih, _whh, _bih, _bhh)
        
    def GRUCell(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy, hy, resetgate, inputgate, newgate

    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            
        self.check_forward_hidden(input, hx, '')
        
        return self.GRUCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
    
class MyEncoderRNN(nn.Module):
    def __init__(self, input_size, _wih, _whh, _bih, _bhh, embedding):
        super(MyEncoderRNN, self).__init__()
        
        # set random seed
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        self.embedding = embedding
        self.gru = MyGRUCell(self.hidden_size, self.embedding_size, _wih=_wih, _whh=_whh, _bih=_bih, _bhh=_bhh)
    
    def forward(self, input, hidden=None):
        output = self.embedding(input).view(1, input.shape[0], -1)
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_size, dtype=output.dtype, device=input.device)
        output, hidden, rg, ig, ng = self.gru(output.view(1, -1), hidden.view(1, -1))
        return output, hidden.unsqueeze(0), rg, ig, ng
    
    def get_embedding(self, input):
        return self.embedding(input).view(1, input.shape[0], -1)
    
class MyDecoderRNN(nn.Module):
    def __init__(self, output_size, _wih, _whh, _bih, _bhh, out, use_same_embedding=False):
        super(MyDecoderRNN, self).__init__()
        
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
            
        self.gru = MyGRUCell(self.hidden_size, self.embedding_size, _wih=_wih, _whh=_whh, _bih=_bih, _bhh=_bhh)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.out.weight.data = out.weight.data
        self.out.bias.data = out.bias.data
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        if not self.use_same_embedding:
            output = self.embedding(input).view(1, input.shape[0], -1)
        else:
            output = input
            
        output, hidden, rg, ig, ng = self.gru(output.view(1, -1), hidden.view(1, -1))
        output = self.softmax(self.out(output))
        
        return output, hidden.unsqueeze(0), rg, ig, ng
    
def text2batch(text):
    '''
    Args:
        text: a list of text
    Return:
        batch: dict with key of sentence and sentence_len
               sentence - torch of size (batch/1, length)
               sentence_len - list of length (batch/1)
    '''
    global word_dict
    
    batch = {}
    batch['sentence'] = torch.tensor([word_dict[t] for t in ['<SOS>'] + text.split() + ['<EOS>']]).unsqueeze(0)
    batch['sentence_len'] = [len(text.split())]
    return batch

def predict2sentence(batch, pred):
    '''
    Args:
        batch: tensor (seq_len, batch_size)
        pred: list (seq_len, batch_size)
    '''
    global idx2word
    
    _gt = batch.cpu().detach().numpy().transpose()
    _pd = np.array(pred).reshape(-1, batch.shape[1]).transpose()
    
    _ls = []
    for g, p in zip(_gt, _pd):
        s1 = ' '.join([idx2word[idx] for idx in g])
        s2 = ' '.join([idx2word[idx] for idx in p])
        _ls.append((s1, s2))
    
    return _ls

def predictWithAutoencoderHiddenAndGates(batch, encoder2, decoder2):
    global use_same_embedding
    global device
    global word_dict
    
    with torch.no_grad():
        batch_size = batch['sentence'].shape[0]
        seq_len = batch['sentence'].shape[1] - 1
        dim = hidden_size
        
        batch['sentence'] = batch['sentence'].to(device).transpose(0, 1) # seq_len, batch
        batch['sentence_len'] = torch.tensor(batch['sentence_len'], device=device)
        encoder_hn = [None]
        
        encoder_hn_dict = []
        encoder_r_gates = []
        encoder_i_gates = []
        encoder_n_gates = []
        
        for x in batch['sentence'][1:]:
            _, hn, _er, _ei, _en = encoder2(x, encoder_hn[-1])
            if len(encoder_hn_dict) == 0:
                encoder_hn_dict.append(np.zeros(hn.shape[-1]).astype(np.float32))
                
            if len(encoder_r_gates) == 0:
                encoder_r_gates.append(np.zeros(_er.shape[-1]).astype(np.float32))
                
            if len(encoder_i_gates) == 0:
                encoder_i_gates.append(np.zeros(_ei.shape[-1]).astype(np.float32))
                
            if len(encoder_n_gates) == 0:
                encoder_n_gates.append(np.zeros(_en.shape[-1]).astype(np.float32))
                
            encoder_hn_dict.append(hn.clone().cpu().detach().numpy().reshape(-1))
            encoder_r_gates.append(_er.clone().cpu().detach().numpy().reshape(-1))
            encoder_i_gates.append(_ei.clone().cpu().detach().numpy().reshape(-1))
            encoder_n_gates.append(_en.clone().cpu().detach().numpy().reshape(-1))
            encoder_hn.append(hn)
        
        encoder_hn.pop(0)
        encoder_hn = torch.stack(encoder_hn, dim=0).squeeze(1).transpose(0, 1) # batch, seq_len, dim
        
        idx = batch['sentence_len'].unsqueeze(1).repeat(1, hidden_size).view(batch_size, 1, hidden_size)
        decoder_hidden = torch.gather(encoder_hn, 1, idx).transpose(0, 1)
        
        decoder_input = torch.tensor([[word_dict['<SOS>']]] * batch_size, device=device)
        decoded_sentence = [decoder_input.squeeze().cpu().detach().numpy()]
        
        _loss = 0
        _acc = torch.ones(batch_size, dtype=torch.uint8, device=device)
        
        decoder_hn_dict = [decoder_hidden.clone().cpu().detach().numpy().reshape(-1)]
        decoder_r_gates = [encoder_r_gates[-1]]
        decoder_i_gates = [encoder_i_gates[-1]]
        decoder_n_gates = [encoder_n_gates[-1]]

        for di in range(seq_len):
            if use_same_embedding:
                decoder_input = encoder2.get_embedding(decoder_input).detach()

            decoder_output, decoder_hidden, _dr, _di, _dn = decoder2(decoder_input, decoder_hidden)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()
            decoder_hn_dict.append(decoder_hidden.clone().cpu().detach().numpy().reshape(-1))
            decoder_r_gates.append(_dr.clone().cpu().detach().numpy().reshape(-1))
            decoder_i_gates.append(_di.clone().cpu().detach().numpy().reshape(-1))
            decoder_n_gates.append(_dn.clone().cpu().detach().numpy().reshape(-1))
            decoded_sentence.append(decoder_input.squeeze().cpu().detach().numpy())

            _loss += F.nll_loss(decoder_output, batch['sentence'][di+1], ignore_index=word_dict['<PAD>'])

            _acc *= torch.clamp((decoder_input.squeeze().data == batch['sentence'][di+1].data) + \
                                 (batch['sentence'][di+1].data == word_dict['<PAD>']), min=0, max=1)
        
        return _acc, _loss/seq_len, decoded_sentence, np.array(encoder_hn_dict), np.array(decoder_hn_dict), \
                np.array(encoder_r_gates), np.array(encoder_i_gates), np.array(encoder_n_gates), \
                np.array(decoder_r_gates), np.array(decoder_i_gates), np.array(decoder_n_gates)
        
def getModel():
    global hidden_size
    global embedding_size
    global use_same_embedding
    global saved_model_path
    
    encoder = EncoderRNN(vocab_size, hidden_size, embedding_size)
    decoder = DecoderRNN(hidden_size, vocab_size, embedding_size, use_same_embedding)

    saved_model = torch.load(saved_model_path)
    encoder.load_state_dict(saved_model['encoder'])
    decoder.load_state_dict(saved_model['decoder'])

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()
    
    # Copy weight from encoder's gru
    for name, param in encoder.named_parameters():
        if param.requires_grad and name in ['gru.weight_ih_l0', 'gru.weight_hh_l0', 'gru.bias_ih_l0', 'gru.bias_hh_l0']:
            if name == 'gru.weight_ih_l0':
                _ewih = param.detach().cpu().numpy()
            elif name == 'gru.weight_hh_l0':
                _ewhh = param.detach().cpu().numpy()
            elif name == 'gru.bias_ih_l0':
                _ebih = param.detach().cpu().numpy()
            elif name == 'gru.bias_hh_l0':
                _ebhh = param.detach().cpu().numpy()
            print(name, param.shape)
            
    # Copy weight from decoder's gru
    for name, param in decoder.named_parameters():
        if param.requires_grad and name in ['gru.weight_ih_l0', 'gru.weight_hh_l0', 'gru.bias_ih_l0', 'gru.bias_hh_l0']:
            if name == 'gru.weight_ih_l0':
                _dwih = param.detach().cpu().numpy()
            elif name == 'gru.weight_hh_l0':
                _dwhh = param.detach().cpu().numpy()
            elif name == 'gru.bias_ih_l0':
                _dbih = param.detach().cpu().numpy()
            elif name == 'gru.bias_hh_l0':
                _dbhh = param.detach().cpu().numpy()
            print(name, param.shape)
    
    encoder2 = MyEncoderRNN(encoder.input_size, _ewih, _ewhh, _ebih, _ebhh, encoder.embedding)
    encoder2 = encoder2.to(device)

    decoder2 = MyDecoderRNN(decoder.output_size, _dwih, _dwhh, _dbih, _dbhh, decoder.out, True)
    decoder2 = decoder2.to(device)

    return encoder2.eval(), decoder2.eval()
