import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from model.parameters import Parameters
from model.paraphraser import Paraphraser

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--model-name', default='', metavar='MN',
                        help='name of model to save (default: "")')
    parser.add_argument('--input-file', default='data/quora/test.csv', metavar='IF',
                        help='name of file with input phrases (default: "data/quora/test.csv")')
    parser.add_argument('--output-file', default='out.txt', metavar='OF',
                    help='name of output file (default: "out.txt")')
    parser.add_argument('--seq-len', default=30, metavar='SL',
                    help='max length of sequence (default: 30)')

    args = parser.parse_args()

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    paraphraser = Paraphraser(parameters)
    paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_' + args.model_name))
    if args.use_cuda:
        paraphraser = paraphraser.cuda()

    result = []
    target = []

    i = 0
    while True:
        next_batch = batch_loader.next_batch_from_file(batch_size=1,
         file_name=args.input_file, return_sentences=True)
        
        if next_batch is None:
            break

        input, sentences = next_batch
        result += [paraphraser.sample_with_input(batch_loader, args.seq_len, use_cuda, input)]
        target += [sentences[1][0]]
        if i % 1000 == 0:
            print(i)
            print('source : ', ' '.join(sentences[0][0]))
            print('target : ', ' '.join(sentences[1][0]))
            print('sampled : ', result[-1])

    np.save('logs/sampled_out.txt', np.array(result))
    np.save('logs/target_out.txt', np.array(target))
    print('------------------------------')
    print('END')








            