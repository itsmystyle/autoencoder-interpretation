import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
import pandas as pd
from preprocessor import Preprocessor

def gen_data(abstract):
    df = {'id':[], 'sentences':[], 'labels':[]}
    label_ls = []
    sentence_ls = []
    
    df['id'].append('### Random ID')
    df['sentences'].append(abstract)
    df['labels'].append('PAD')
    
    dataframe = pd.DataFrame(df)
    
    return dataframe

def main(args):
    # load config
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    # load embedding
    logging.info('loading embedding...')
    with open(config['model_parameters']['embedding'], 'rb') as f:
        embedding = pickle.load(f)
        config['model_parameters']['embedding'] = embedding.vectors

    # make model
    if config['arch'] == 'Predictor':
        from predictor import Predictor
        PredictorClass = Predictor
        
    predictor = PredictorClass(metrics=[], word_dict=embedding.word_dict,
                               **config['model_parameters'])
    model_path = os.path.join(
        args.model_dir,
        'model.pkl.{}'.format(args.epoch))
    logging.info('loading model from {}'.format(model_path))
    predictor.load(model_path)
    
    if args.input_mode == 0:
        preprocessor = Preprocessor(None)
        preprocessor.embedding = embedding
        
        while True:
            test = input("Enter abstract: (type q to quit)")
            if test == 'q':
                break

    #         test = 'To investigate the efficacy of @ weeks of daily low-dose oral prednisolone in improving pain , mobility , and systemic low-grade inflammation in the short term and whether the effect would be sustained at @ weeks in older adults with moderate to severe knee osteoarthritis ( OA ) .$$$A total of @ patients with primary knee OA were randomized @:@ ; @ received @ mg/day of prednisolone and @ received placebo for @ weeks .$$$Outcome measures included pain reduction and improvement in function scores and systemic inflammation markers .$$$Pain was assessed using the visual analog pain scale ( @-@ mm ) .$$$Secondary outcome measures included the Western Ontario and McMaster Universities Osteoarthritis Index scores , patient global assessment ( PGA ) of the severity of knee OA , and @-min walk distance ( @MWD ) .$$$Serum levels of interleukin @ ( IL-@ ) , IL-@ , tumor necrosis factor ( TNF ) - , and high-sensitivity C-reactive protein ( hsCRP ) were measured .$$$There was a clinically relevant reduction in the intervention group compared to the placebo group for knee pain , physical function , PGA , and @MWD at @ weeks .$$$The mean difference between treatment arms ( @ % CI ) was @ ( @-@ @ ) , p < @ ; @ ( @-@ @ ) , p < @ ; @ ( @-@ @ ) , p < @ ; and @ ( @-@ @ ) , p < @ , respectively .$$$Further , there was a clinically relevant reduction in the serum levels of IL-@ , IL-@ , TNF - , and hsCRP at @ weeks in the intervention group when compared to the placebo group .$$$These differences remained significant at @ weeks .$$$The Outcome Measures in Rheumatology Clinical Trials-Osteoarthritis Research Society International responder rate was @ % in the intervention group and @ % in the placebo group ( p < @ ) .$$$Low-dose oral prednisolone had both a short-term and a longer sustained effect resulting in less knee pain , better physical function , and attenuation of systemic inflammation in older patients with knee OA ( ClinicalTrials.gov identifier NCT@ ) .'

            df = gen_data(test)
            inpt = preprocessor.get_dataset(df, 1)
            predicts = predictor.predict_dataset(inpt, inpt.collate_fn)

            print_predict(predicts, test)
        
    else:
        if args.input_dir is None or args.output_dir is None:
            print('Please set input and output directory path.')
            
        logging.info('loading test data...')
        with open(args.input_dir, 'rb') as f:
            test = pickle.load(f)
            test.shuffle = False
        logging.info('predicting...')
        predicts = predictor.predict_dataset(test, test.collate_fn)

        output_path = os.path.join(args.output_dir)
        write_predict(predicts, test, output_path)

def print_predict(predicts, data):
    labels_dict = {0: 'BACKGROUND', 1: 'OBJECTIVE', 2: 'METHODS', 3: 'RESULTS', 4: 'CONCLUSIONS', 5: 'PAD'}
    
    print('===================================================================================')
    labels = predicts[0][1][0].tolist()
    for idx, sentence in enumerate(data.split('$$$')):
        print(labels_dict[labels[idx]] + '---> ' + sentence)
    print('========================================END========================================')
    
def write_predict(predicts, data, output_path):
    labels_dict = {0: 'BACKGROUND', 1: 'OBJECTIVE', 2: 'METHODS', 3: 'RESULTS', 4: 'CONCLUSIONS', 5: 'PAD'}
    
    with open(output_path, 'w') as fout:
        for preds in predicts:
            ids = preds[0]
            labels = preds[1].tolist()
            sent_len = preds[2]

            for x, y, z in zip(ids, labels, sent_len):
                fout.write(x + '\n' + ' '.join([labels_dict[l] for l in y[:z]]) + '\n\n')

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--input_mode', type=int, default=0, help='Input mode. 0: type yourself, 1: read from text file.')
    parser.add_argument('--input_dir', type=str,
                        help='input path')
    parser.add_argument('--output_dir', type=str,
                        help='output path')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--epoch', type=int, default=7)
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
        pdb.post_mortem(tb)
