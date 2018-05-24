import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from model.parameters import Parameters
from model.paraphraser import Paraphraser

def sample_with_input_file(batch_loader, paraphraser, args, input_file):
    result, target, i = [], [] , 0
    while True:
        next_batch = batch_loader.next_batch_from_file(batch_size=1,
         file_name=input_file, return_sentences=True)
        
        if next_batch is None:
            break

        input, sentences = next_batch
        input = [var.cuda() if args.use_cuda else var for var in input]

        result += [paraphraser.sample_with_input(batch_loader,
                                 args.seq_len, 
                                 args.use_cuda,
                                 args.use_mean,
                                input)]
        target += [sentences[1][0]]
        if i % 1000 == 0:
            print(i)
            print('source : ', ' '.join(sentences[0][0]))
            print('target : ', ' '.join(sentences[1][0]))
            print('sampled : ', result[-1])
        i += 1
    return result, target

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--model-name', default='', metavar='MN',
                        help='name of model to save (default: "")')
    parser.add_argument('--input-file', default='quora_test', metavar='IF',
                        help='name of file with input phrases (default: "quora_test")')
    parser.add_argument('--output-file', default='out.txt', metavar='OF',
                    help='name of output file (default: "out.txt")')
    parser.add_argument('--use-mean', type=bool, default=False, metavar='UM',
                    help='use mean value instead of sampling z')
    parser.add_argument('--seq-len', default=30, metavar='SL',
                    help='max length of sequence (default: 30)')
    parser.add_argument('--use-quora', default=False, type=bool, metavar='quora', 
                    help='if include quora dataset (default: True)')
    parser.add_argument('--use-snli', default=False, type=bool, metavar='snli', 
                    help='if include snli dataset (default: True)')
    parser.add_argument('--use-coco', default=False, type=bool, metavar='coco', 
                    help='if include mscoco dataset (default: False)')

    args = parser.parse_args()
    datasets = set()
    if args.use_quora is True:
        datasets.add('quora')
    if args.use_snli is True:
        datasets.add('snli')
    if args.use_coco is True:
        datasets.add('mscoco')

    print('use mean' , args.use_mean)

    batch_loader = BatchLoader(datasets=datasets)
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    paraphraser = Paraphraser(parameters)
    paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_' + args.model_name))
    if args.use_cuda:
        paraphraser = paraphraser.cuda()

    result, target = sample_with_input_file(batch_loader, paraphraser, args, args.input_file)

    if args.input_file not in ['snli_test', 'mscoco_test', 'quora_test', 'snips']:
        args.input_file = 'custom_file'

    sampled_file_dst = 'logs/sampled_out_{}_{}{}.txt'.format(args.input_file,
                                            'mean_' if args.use_mean else '', args.model_name)
    target_file_dst = 'logs/target_out_{}_{}{}.txt'.format(args.input_file,
                                            'mean_' if args.use_mean else '', args.model_name)
    np.save(sampled_file_dst, np.array(result))
    np.save(target_file_dst, np.array(target))
    print('------------------------------')
    print('results saved to: ')
    print(sampled_file_dst)
    print(target_file_dst)
    print('END')








            
