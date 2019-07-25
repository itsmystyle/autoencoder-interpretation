import torch
import torch.nn.functional as F
import pdb
from sklearn.metrics import precision_recall_fscore_support

class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass

class Accuracy(Metrics):
    """
    Accuracy a.k.a Micro F1 score
    """
    def __init__(self):
        self.n = 0
        self.n_corrects = 0
        self.name = 'Accuracy'

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, logits, batch):
        """
        Args:
            predicts (Tuple): (logits, labels) - with size (batch * n_sent_len, output_tag).
            batch (dict): batch. ['sentences', 'sentences_len', 'labels']
        """
        self.n_corrects += torch.sum(logits).item()
        self.n += batch['sentence_len'].shape[0]

    def get_score(self):
        return self.n_corrects / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)
