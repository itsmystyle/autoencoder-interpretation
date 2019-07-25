import torch
import torch.nn.functional as F
from base_predictor import BasePredictor
from modules import *
import pdb

class Predictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self,
                 optimizer,
                 hidden_size,
                 word_dict=None,
                 use_teacher_forcing=False,
                 encoder_model_name="EncoderNet",
                 decoder_model_name="DecoderNet",
                 weight_decay=0,
                 **kwargs):
        super(Predictor, self).__init__(**kwargs)
        
        self.use_teacher_forcing = use_teacher_forcing
        self.word_dict = word_dict
        self.vocab_size = len(word_dict)
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        
        if encoder_model_name == "EncoderRNN":
            self.encoder = EncoderRNN(self.vocab_size, self.hidden_size)
        
        if decoder_model_name == "DecoderRNN":
            self.decoder = DecoderRNN(self.hidden_size, self.vocab_size)

        # use cuda
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        # make optimizer
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                                              lr=self.learning_rate, 
                                              weight_decay=weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                                             lr=self.learning_rate, 
                                             momentum=0.9, 
                                             weight_decay=weight_decay)

    def _run_iter(self, batch, training):
        batch_size = batch['sentence'].shape[0]
        seq_len = batch['sentence'].shape[1] - 1
        dim = self.hidden_size
        
        batch['sentence'] = batch['sentence'].to(self.device).transpose(0, 1) # seq_len, batch
        batch['sentence_len'] = torch.tensor(batch['sentence_len'], device=self.device)
        encoder_hn = [None]
        
        for x in batch['sentence'][1:]:
            _, hn = self.encoder(x, encoder_hn[-1])
            encoder_hn.append(hn)
        
        encoder_hn.pop(0)
        encoder_hn = torch.stack(encoder_hn, dim=0).squeeze(1).transpose(0, 1) # batch, seq_len, dim
        
        idx = batch['sentence_len'].unsqueeze(1).repeat(1, self.hidden_size).view(batch_size, 1, self.hidden_size)
        decoder_hidden = torch.gather(encoder_hn, 1, idx).transpose(0, 1)
        
        decoder_input = torch.tensor([[self.word_dict['<SOS>']]] * batch_size, device=self.device)
        
        if self.use_teacher_forcing:
            pass
        
        else:
            
            loss = 0
            acc = torch.ones(batch_size, dtype=torch.uint8, device=self.device)
            
            for di in range(seq_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, 
                                                              decoder_hidden)
                
                
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()

                loss += F.nll_loss(decoder_output, batch['sentence'][di+1], ignore_index=self.word_dict['<PAD>'])
                
                acc *= torch.clamp((decoder_input.squeeze().data == batch['sentence'][di+1].data) + (batch['sentence'][di+1].data == self.word_dict['<PAD>']), min=0, max=1)
        
        return acc, loss/seq_len

    def _predict_batch(self, batch):
        pass