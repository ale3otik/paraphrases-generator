import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, params, highway):
        super(Encoder, self).__init__()

        self.params = params
        self.hw1 = highway
        
        # encoding source and target
        self.rnns = nn.ModuleList([nn.LSTM(input_size=self.params.word_embed_size,
                                       hidden_size=self.params.encoder_rnn_size,
                                       num_layers=self.params.encoder_num_layers,
                                       batch_first=True,
                                       bidirectional=True) for i in range(2)])

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 4, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 4, self.params.latent_variable_size)

    def forward(self, input_source, input_target):
        """
        :param input_source: [batch_size, seq_len, embed_size] tensor
        :param input_target: [batch_size, seq_len, embed_size] tensor
        :return: distributinon parameters of input sentenses with shape of 
            [batch_size, latent_variable_size]
        """

        # (num_layers * num_directions, batch, hidden_size)        
        state = None
        for i, input in enumerate([input_source , input_target]):
            [batch_size, seq_len, embed_size] = input.size()

            input = input.view(-1, embed_size)
            input = self.hw1(input)
            input = input.view(batch_size, seq_len, embed_size)

            _, state = self.rnns[i](input, state)

        [h_state, c_state] = state
        h_state = h_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]
        c_state = c_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]
        h_state = h_state.permute(1,0,2).contiguous().view(batch_size, -1)
        c_state = c_state.permute(1,0,2).contiguous().view(batch_size, -1)
        final_state = t.cat([h_state, c_state], 1)

        mu, logvar = self.context_to_mu(final_state), self.context_to_logvar(final_state)
        return mu, logvar
