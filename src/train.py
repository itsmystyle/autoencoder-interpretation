import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from callbacks import ModelCheckpoint, MetricsLogger
from metrics import *
from lr_finder import LRFinder


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
        
    logging.info('loading word dictionary...')
    with open(config['words_dict'], 'rb') as f:
        words_dict = pickle.load(f)

    logging.info('loading train data...')
    with open(config['train'], 'rb') as f:
        train = pickle.load(f)
        
    logging.info('loading validation data...')
    with open(config['model_parameters']['valid'], 'rb') as f:
        valid = pickle.load(f)
    config['model_parameters']['valid'] = valid
    
    if args.lr_finder:
#         logging.info('creating model!')
        
#         vocab_size = len(words_dict)
#         hidden_size = config['model_parameters']['hidden_size']
#         embedding_size = config['model_parameters']['embedding_size']
#         use_same_embedding = config['model_parameters']['use_same_embedding']
        
#         encoder = EncoderRNN(vocab_size, hidden_size, embedding_size)
#         decoder = DecoderRNN(hidden_size, vocab_size, embedding_size, use_same_embedding)
        
#         criterion = nn.CrossEntropyLoss()
        
#         if config['model_parameters']['optimizer'] == 'Adam':
#             optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
#                                               lr=1e-7, 
#                                               weight_decay=1e-2)
#         elif config['model_parameters']['optimizer'] == 'SGD':
#             optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), 
#                                              lr=1e-7, 
#                                              momentum=0.9, 
#                                              weight_decay=1e-2)
        
#         lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
#         lr_finder.range_test(trainloader, end_lr=100, num_iter=100, step_mode="exp")
        
#         # use cuda
#         self.encoder = self.encoder.to(self.device)
#         self.decoder = self.decoder.to(self.device)
        pass
        
    else:
        if config['arch'] == 'Predictor':
            from predictor import Predictor
            PredictorClass = Predictor

        predictor = PredictorClass(
            metrics=[Accuracy()], word_dict=words_dict,
            **config['model_parameters']
        )

        metrics_logger = MetricsLogger(
            os.path.join(args.model_dir, 'log.json')
        )

        if args.load is not None:
            predictor.load(args.load)
            try:
                metrics_logger.load(int(args.load.split('.')[-1]))
            except:
                metrics_logger.load(448)

        model_checkpoint = ModelCheckpoint(
            os.path.join(args.model_dir, 'model.pkl'),
            'Accuracy', 1, 'max'
        )

        logging.info('start training!')
        predictor.fit_dataset(train,
                              train.collate_fn,
                              [model_checkpoint, metrics_logger])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--lr_finder', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
