import torch
from tqdm import tqdm


class Metrics:
    def __init__(self):
        self.name = "Metric Name"

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Accuracy(Metrics):
    """
    Accuracy
    """

    def __init__(self):
        self.n = 0
        self.n_corrects = 0
        self.name = "Accuracy"

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
        self.n += batch["sentence_len"].shape[0]

    def get_score(self):
        return self.n_corrects / self.n

    def print_score(self):
        score = self.get_score()
        return "{:.5f}".format(score)


class WordAccuracy(Metrics):
    """
    Words Accuracy
    """
    class Word:
        """
        Accuracy of a word
        """
        def __init__(self):
            # n_predict is the number of word appear
            # n_correct is the number of word that doesnt appear
            # n is the number of word in dataset

            self.n_predict = 0
            self.n_correct = 0
            self.n = 0

        def get_score(self):
            return self.n_correct / self.n

        def get_predict_rate(self):
            return self.n_predict / self.n

    def __init__(self, dataloader, idx2word):
        # word_dict[word] = Word()
        self.reset(dataloader, idx2word)
        self.name = "Word Acc."

    def reset(self, dataloader, idx2word, verbose=True):
        self.word_dict = {}
        self.idx2word = idx2word

        if verbose:
            trange = tqdm(dataloader, total=len(dataloader))
        else:
            trange = dataloader

        for datas in trange:
            for data in datas["sentence"]:
                for idx in data[1:]:
                    word = self.idx2word[idx.item()]
                    if word == "<PAD>":
                        break
                    if word not in self.word_dict:
                        self.word_dict[word] = self.Word()
                    self.word_dict[word].n += 1

    def update(self, logits, batch):
        """
        Args:
            logits (batch_size, sequence_length, 1)
            batch ['sentence', 'sentence_len'] (sequence_length, batch_size)
            ** sequence_length = <SOS> + sentence_len + <EOS> + n*<PAD>
        """
        for pred, ground, len in zip(
            logits.transpose(),
            batch["sentence"].cpu().detach().numpy().transpose(),
            batch["sentence_len"].cpu().detach().numpy(),
        ):
            for idx in range(1, len + 2):
                # n_correct +1
                _word = self.idx2word[pred[idx]]
                if _word not in self.word_dict:
                    self.word_dict[_word] = self.Word()
                    self.word_dict[_word].n += 1

                if _word == self.idx2word[ground[idx]]:
                    self.word_dict[_word].n_correct += 1

                # n_predict +1
                self.word_dict[_word].n_predict += 1

    def get_score_dict(self):
        # Return word dictionary
        return self.word_dict

    def get_word(self, word):
        return self.word_dict[word]

    def print_score(self):
        # Print avg. accuracy
        score = [v.get_score() for k, v in self.word_dict.items()]
        return "{:.5f}".format(sum(score) / len(score))
