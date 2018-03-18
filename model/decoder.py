import torch as t
import torch.nn as nn
import torch.nn.functional as F

from utils.functional import parameters_allocation_check


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.params = params
        
        self.encoding_rnn = nn.LSTM(input_size=self.params.word_embed_size,
                                       hidden_size=self.params.decoder_rnn_size,
                                       num_layers=self.params.decoder_num_layers,
                                       batch_first=True,
                                       bidirectional=True)
        
        self.decoding_rnn = nn.LSTM(input_size=self.params.latent_variable_size + self.params.word_embed_size,
                                       hidden_size=self.params.decoder_rnn_size,
                                       num_layers=self.params.decoder_num_layers,
                                       batch_first=True)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

    def build_initial_state(self, input):
        [batch_size, seq_len, embed_size] = input.size()
        input = input.view(-1, embed_size)
        input = self.hw1(input)
        input = input.view(batch_size, seq_len, embed_size)

        _, cell_state = self.encoding_rnn(input)

        return cell_state


    def forward(self, encoder_input, decoder_input, z, drop_prob, initial_state=None):
        """
        :param encoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn

        :return: unnormalized logits of sentense words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """
        
        if initial_state is None:
            # build initial context with source input.
            assert encoder_input not is None
            initial_state = self.build_initial_state(encoder_input)

        [batch_size, seq_len, _] = decoder_input.size()

        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.decoding_rnn(decoder_input, initial_state)

        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state
