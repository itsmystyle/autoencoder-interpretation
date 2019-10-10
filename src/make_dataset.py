import os
import pdb
import sys
import json
import logging
import traceback
import collections

import argparse
import pickle
from tqdm import tqdm

from preprocessor import Preprocessor


def main(args):
    config_path = os.path.join(args.dest_dir, "config.json")
    logging.info("loading configuration from {}".format(config_path))
    with open(config_path) as f:
        config = json.load(f)

    preprocessor = Preprocessor(None)

    logging.info("loading training data from {}".format(config["train_path"]))
    with open(config["train_path"], "r") as f:
        train_data = f.readlines()

    logging.info("loading validation data from {}".format(config["valid_path"]))
    with open(config["valid_path"], "r") as f:
        valid_data = f.readlines()

    logging.info("loading testing data from {}".format(config["test_path"]))
    with open(config["test_path"], "r") as f:
        test_data = f.readlines()

    logging.info(
        "loading long corpus testing data from {}".format(config["long_test_path"])
    )
    with open(config["long_test_path"], "r") as f:
        long_test_data = f.readlines()

    # collect words appear in the data
    logging.info("collecting words from training set...")
    words = collections.Counter()
    for data in tqdm(train_data, total=len(train_data)):
        words.update(data.strip().split())
    logging.info("{} words collected".format(len(words)))

    # build sorted vocab dictionary (for adaptive_softmax loss later)
    word_dict = {}
    counter = 4
    word_dict["<PAD>"] = 0
    word_dict["<UNK>"] = 1
    word_dict["<SOS>"] = 2
    word_dict["<EOS>"] = 3
    for word in tqdm(words.most_common()):
        if word[1] > args.threshold:
            word_dict[word[0]] = counter
            counter += 1

    logging.info("{} words saved".format(counter))

    vocab_path = "_{}.pkl".format(args.threshold).join(
        config["vocab_path"].split(".pkl")
    )
    word_set_path = "_{}.pkl".format(args.threshold).join(
        config["word_set_path"].split(".pkl")
    )

    with open(vocab_path, "wb") as fout:
        pickle.dump(word_dict, fout)
    with open(word_set_path, "wb") as fout:
        pickle.dump(words, fout)
    logging.info(
        "Word frequency and vocab saved in {}, {}".format(word_set_path, vocab_path)
    )

    # update word dictionary used by preprocessor
    preprocessor.words_dict = word_dict

    # train
    logging.info("Processing training set from {}".format(config["train_path"]))
    train = preprocessor.get_dataset(train_data, args.n_workers)
    train_pkl_path = os.path.join(
        args.dest_dir, "train_{}.pkl".format(args.threshold)
    )
    logging.info("Saving training set to {}".format(train_pkl_path))
    with open(train_pkl_path, "wb") as f:
        pickle.dump(train, f)

    # valid
    logging.info("Processing validation set from {}".format(config["valid_path"]))
    valid = preprocessor.get_dataset(valid_data, args.n_workers)
    valid_pkl_path = os.path.join(
        args.dest_dir, "valid_{}.pkl".format(args.threshold)
    )
    logging.info("Saving validation set to {}".format(valid_pkl_path))
    with open(valid_pkl_path, "wb") as f:
        pickle.dump(valid, f)

    # test
    logging.info('Processing testing set from {}'.format(config['test_path']))
    test = preprocessor.get_dataset(test_data, args.n_workers)
    test_pkl_path = os.path.join(args.dest_dir, 'test_{}.pkl'.format(args.threshold))
    logging.info('Saving testing set to {}'.format(test_pkl_path))
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test, f)

    # long test
    logging.info('Processing long corpus testing set from {}'.format(config['long_test_path']))
    long_test = preprocessor.get_dataset(long_test_data, args.n_workers)
    long_test_pkl_path = os.path.join(args.dest_dir, 'long_test_{}.pkl'.format(args.threshold))
    logging.info('Saving long corpus testing set to {}'.format(long_test_pkl_path))
    with open(long_test_pkl_path, 'wb') as f:
        pickle.dump(long_test, f)

# TEST
#     dataloader = DataLoader(long_test,
#                             collate_fn=long_test.collate_fn,
#                             batch_size=4,
#                             shuffle=False, num_workers=args.n_workers)

#     for data in dataloader:
#         import pdb
#         pdb.set_trace()
#         print(data)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle."
    )
    parser.add_argument(
        "dest_dir", type=str, help="[input] Path to the directory that ."
    )
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        level=logging.INFO,
        datefmt="%m-%d %H:%M:%S",
    )
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
