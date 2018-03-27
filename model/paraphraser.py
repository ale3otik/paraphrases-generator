import math
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .decoder import Decoder
from .encoder import Encoder
from .highway import Highway

class Paraphraser(nn.Module):
    def __init__(self, params):
        super(Paraphraser, self).__init__()
        self.params = params
        self.highway = Highway(self.params.word_embed_size, 2, F.relu)
        self.encoder = Encoder(self.params, self.highway)
        self.decoder = Decoder(self.params, self.highway)

    def forward(self, drop_prob, encoder_input=None, decoder_input=None, 
        z=None, initial_state=None, use_cuda=True):
        """
        :param encoder_word_input: An list of 2 tensors with shape of [batch_size, seq_len] of Long type
        :param decoder_word_input: An An list of 2 tensors with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            [batch_size, _, _] = encoder_input[0].size()

            mu, logvar = self.encoder(encoder_input[0], encoder_input[1])
            std = t.exp(0.5 * logvar)
            
            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()
            z = z * std + mu

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None

        out, final_state = self.decoder(decoder_input[0], decoder_input[1],
                                        z, drop_prob, initial_state)
        return out, final_state, kld

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_input_source, 
             encoder_input_target, 
             decoder_input_source, 
             decoder_input_target, target] = input

            logits, _, kld = self(dropout, 
                    (encoder_input_source, encoder_input_target),
                    (decoder_input_source, decoder_input_target), 
                    z=None, use_cuda=use_cuda)

            logits = logits.view(-1, self.params.vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            loss = self.params.cross_entropy_penalty_weight * cross_entropy \
                +  self.params.get_kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, kld, self.params.get_kld_coef(i)

        return train

    def validater(self, batch_loader):
        def get_samples(logits, target):
            '''
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            '''

            ## for version > 0.4
            # prediction = F.softmax(logits, dim=-1).data.cpu().numpy()
            
            ## for version < 0.3
            seq_len = logits.size()[1]
            prediction = F.softmax(
                logits.view(-1, self.params.vocab_size)).view(-1, seq_len, self.params.vocab_size)
            prediction = prediction.data.cpu().numpy()
            
            target = target.data.cpu().numpy()

            sampled, expected = [], []
            for i in range(prediction.shape[0]):
                sampled  += [' '.join([batch_loader.sample_word_from_distribution(d) 
                    for d in prediction[i]])]
                expected += [' '.join([batch_loader.get_word_by_idx(idx) for idx in target[i]])]

            return sampled, expected


        def validate(batch_size, use_cuda, need_samples=False):
            if need_samples:
                input, sentences = batch_loader.next_batch(batch_size, 'test', return_sentences=True)
                sentences = [' '.join(s) for s in sentences]
            else:
                input = batch_loader.next_batch(batch_size, 'test')
            
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_input_source, 
             encoder_input_target, 
             decoder_input_source, 
             decoder_input_target, target] = input

            logits, _, kld = self(0., (encoder_input_source, encoder_input_target),
                                    (decoder_input_source, decoder_input_target), 
                                    z=None, use_cuda=use_cuda) 

            if need_samples:
                [s1, s2] = sentences
                sampled, _ = get_samples(logits, target)
            else:
                s1, s2 = (None, None)
                sampled, _ = (None, None)

            logits = logits.view(-1, self.params.vocab_size)
            target = target.view(-1)

            cross_entropy = F.cross_entropy(logits, target)

            return cross_entropy, kld, (sampled, s1, s2)

        return validate

    # def sample_from_pair(self,  batch_loader, seq_len, seed, use_cuda, encoder_input=None):
    #     decoder_word_input_np,  = batch_loader.go_input(1)

    #     # print('decoder word input : ', decoder_word_input_np)

    #     decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
    #     decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())
    #     if use_cuda:
    #         decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

    #     result = ''

    #     initial_state = None

    #     for i in range(seq_len):
    #         logits, initial_state, _ = self(0., encoder_word_input, encoder_character_input,
    #                                         decoder_word_input, decoder_character_input,
    #                                         seed, initial_state)

    #         logits = logits.view(-1, self.params.word_vocab_size)
    #         prediction = F.softmax(logits)

    #         word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])
    #         # word = batch_loader.sample_most_pvirobable_word(prediction.data.cpu().numpy()[-1])

    #         if word == batch_loader.end_token:
    #             break

    #         result += ' ' + word

    #         decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
    #         decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

    #         decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
    #         decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

    #         if use_cuda:
    #             decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

    #     return result

    # def conditioned_sample(self, input_phrase, batch_loader, args):
    #     encoder_word_input_np = batch_loader.sentence_to_word_input(input_phrase)
    #     encoder_character_input_np = batch_loader.sentence_to_character_input(input_phrase)
    #     encoder_word_input = Variable(t.from_numpy(encoder_word_input_np).long())
    #     encoder_character_input = Variable(t.from_numpy(encoder_character_input_np).long())

    #     if args.use_cuda:
    #         encoder_word_input = encoder_word_input.cuda()
    #         encoder_character_input = encoder_character_input.cuda()

    #     return self.sample(batch_loader, 50, None, args.use_cuda, 
    #                             encoder_word_input, encoder_character_input)

