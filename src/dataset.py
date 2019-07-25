import random
import torch
from torch.utils.data import Dataset


class CorpusDataset(Dataset):
    """
    Args:
        data (list): List of samples.
        padding (int): Index used to pad sequences to the same length.
    """
    def __init__(self, data, padding=0, sp_tag=[None, None]):
        self.data = data
        self.padding = padding
        self.SOS_TAG = sp_tag[0]
        self.EOS_TAG = sp_tag[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        # datas : dict
        # sentence
        
        batch = {}
        
        # sentences
        _sentence_len = [len(data['sentence']) for data in datas]
        _sentence_padded_len = max(_sentence_len)
        _sentence = torch.tensor([([self.SOS_TAG] + 
                                   data['sentence'] + 
                                   [self.EOS_TAG] + 
                                   [self.padding]*(_sentence_padded_len - len(data['sentence']))
                                  ) for data in datas])
        
        batch['sentence'] = _sentence
        batch['sentence_len'] = _sentence_len

        return batch

def pad_to_len(arr, padded_len, padding=0):
    """ Pad `arr` to `padded_len` with padding if `len(arr) < padded_len`.
    If `len(arr) > padded_len`, truncate arr to `padded_len`.
    Example:
        pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
        pad_to_len([1, 2, 3, 4, 5, 6], 5, -1) == [1, 2, 3, 4, 5]
    Args:
        arr (list): List of int.
        padded_len (int)
        padding (int): Integer used to pad.
    """
    arr_len = len(arr)
    if arr_len > padded_len:
        return arr[:padded_len-1] + [arr[padded_len]]
    elif arr_len < padded_len:
        return arr + [padding]*(padded_len - arr_len)

    return arr
