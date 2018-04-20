import math

class Parameters:
    def __init__(self, max_seq_len, vocab_size):
        self.max_seq_len = int(max_seq_len) + 1  # go or eos token

        self.vocab_size = int(vocab_size)
        self.word_embed_size = 300 # must be 300 for Glove300 embedding

        self.encoder_rnn_size = 600
        self.encoder_num_layers = 1

        self.latent_variable_size = 1100

        self.decoder_rnn_size = 600
        self.decoder_num_layers = 2

        self.kld_penalty_weight = 1.0
        self.cross_entropy_penalty_weight = 79.0

    def get_kld_coef(self, i):
        return self.kld_penalty_weight * (math.tanh((i - 3500)/1000) + 1)/2.0
