import torch as t
import torch.nn as nn
import torch.nn.functional as F

from highway import Highway
from utils.functional import parameters_allocation_check


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        self.hw1 = Highway(self.params.word_embed_size, 2, F.relu)
        
        # encoding source and target
        self.rnns = nn.ModuleList([nn.LSTM(input_size=self.params.word_embed_size,
                                       hidden_size=self.params.encoder_rnn_size,
                                       num_layers=self.params.encoder_num_layers,
                                       batch_first=True,
                                       bidirectional=True) for i in range(2)])

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

    def forward(self, input_source, input_target):
        """
        :param input_source: [batch_size, seq_len, embed_size] tensor
        :param input_target: [batch_size, seq_len, embed_size] tensor
        :return: distributinon parameters of input sentenses with shape of 
            [batch_size, latent_variable_size]
        """

        # (num_layers * num_directions, batch, hidden_size)        
        h_state, c_state = None, None
        for i, input in enumerate([input_source , input_target]):
            [batch_size, seq_len, embed_size] = input.size()

            input = input.view(-1, embed_size)
            input = self.hw1(input)
            input = input.view(batch_size, seq_len, embed_size)

            _, (h_state, c_state) = self.rnns[i](input, (h_state, c_state))
        
        h_state = h_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]
        c_state = c_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]
        h_state = h_state.permute(1,0,2).view(batch_size, -1)
        c_state = c_state.permute(1,0,2).view(batch_size, -1)
        final_state = t.cat([h_state, c_state], 1)

        mu, logvar = context_to_mu(final_state), context_to_logvar(final_state)
        return mu, logvar
