import json
import logging
from multiprocessing import Pool
from dataset import CorpusDataset
from tqdm import tqdm
import pandas as pd


class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, words_dict):
        self.words_dict = words_dict
        self.logging = logging.getLogger(name=__name__)

    def tokenize(self, sentence):
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            indices (list of str): List of tokens in a sentence.
        """
        return sentence.split()

    def sentence_to_indices(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        return [self.words_dict[word] if word in self.words_dict else self.words_dict["<UNK>"] for word in self.tokenize(sentence)]

    def get_dataset(self, dataset, n_workers=4, dataset_args={}):
        """ Load data and return Dataset objects for training and validating.

        Args:
            data_path (str): Path to the data.
        """
        self.logging.info('preprocessing data...')

        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(dataset) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(dataset)
                else:
                    batch_end = (len(dataset) // n_workers) * (i + 1)

                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples, [batch])

            pool.close()
            pool.join()

        processed = []
        for result in results:
            processed += result.get()

        padding = self.words_dict["<PAD>"]
        sp_tag = [self.words_dict["<SOS>"], self.words_dict["<EOS>"]]
        return CorpusDataset(processed, padding=padding, sp_tag=sp_tag, **dataset_args)

    def preprocess_samples(self, dataset):
        """ Worker function.

        Args:
            dataset (list)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset, total=len(dataset)):
            processed.append(self.preprocess_sample(sample))

        return processed

    def preprocess_sample(self, data):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        processed['sentence'] = self.sentence_to_indices(data.strip())

        return processed
