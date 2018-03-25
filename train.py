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
    parser.add_argument('--num-iterations', type=int, default=60000, metavar='NI',
                        help='num iterations (default: 60000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--model-name', default='', metavar='MN',
                        help='name of model to save (default: '')')
    args = parser.parse_args()

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    paraphraser = Paraphraser(parameters)
    ce_result = []
    kld_result = []

    if args.use_trained:
        paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_' + args.model_name))
        ce_result = list(np.load('logs/ce_result_{}.npy'.format(args.model_name)))
        kld_result = list(np.load('logs/kld_result_npy_{}.npy'.format(args.model_name)))

    if args.use_cuda:
        paraphraser = paraphraser.cuda()

    optimizer = Adam(paraphraser.learnable_parameters(), args.learning_rate)

    train_step = paraphraser.trainer(optimizer, batch_loader)
    validate = paraphraser.validater(batch_loader)

    for iteration in range(args.num_iterations):

        cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

        if iteration % 100 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy.data.cpu().numpy())
            print('-------------KLD--------------')
            print(kld.data.cpu().numpy())
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')

        # validation
        if iteration % 500 == 0:
            # averaging across several batches
            cross_entropy, kld = [], []
            for i in range(5):
                ce, kl, _ = validate(args.batch_size, args.use_cuda)
                cross_entropy += [ce.data.cpu().numpy()]
                kld += [kl.data.cpu().numpy()]
            
            kld = np.mean(kld)
            cross_entropy = np.mean(cross_entropy)
            ce_result += [cross_entropy]
            kld_result += [kld]

            print('\n')
            print('------------VALID-------------')
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy)
            print('-------------KLD--------------')
            print(kld)
            print('------------------------------')

            _, _, (sampled, expected) = validate(2, args.use_cuda, need_samples=True)
            
            for s, e in zip(sampled, expected):
                print('sampled: ' + s)
                print('taret: ' + e)
                print('...........................')

        # save model
        if iteration % 1000 == 0 or iteration == (args.num_iterations - 1):
            t.save(paraphraser.state_dict(), 'saved_models/trained_paraphraser_' + args.model_name)
            np.save('logs/ce_result_{}.npy'.format(args.model_name), np.array(ce_result))
            np.save('logs/kld_result_npy_{}'.format(args.model_name), np.array(kld_result))
            