import collections
import os
import re
import torch as t
import numpy as np
import pandas as pd

class BatchLoader:
    def __init__(self, vocab_size=10000, sentences=None, path='../../'):
        '''
            Build vocab for sentences or for data files in path if None. 
        '''
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_vec = {}
        self.max_seq_len = 0

        self.unk_label = '<unk>'
        self.end_label = '</s>'
        self.go_label = '<s>'

        self.data_files = [path + 'data/quora/train.txt',
                           path + 'data/quora/test.txt']
        self.glove_path = [path + 'data/glove.840B.300d.txt']

        if sentences is None:
            self.data = [pd.read_csv(f)[['question1', 'question2']] for f in data_files[0]]

        self.build_vocab(sentences)


    def clean_str(self, string):
        '''
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        '''
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?:;.\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r":", " : ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def get_encoder_input(self, sentences):
        return [self.embed_batch([s + [self.end_label] for s in q]) for q in sentences]
    
    def get_decoder_input(self, sentences): 
        enc_inp = self.embed_batch([s + [self.end_label] for s in sentences[0]]) 
        dec_inp = self.embed_batch([[self.go_label] + s + [self.end_label] for s in sentences[1]]) 
        return [enc_inp, dec_inp]
    
    def get_target(self, sentences):
        sentences = sentences[1]
        target_idx = [[get_idx_by_word(w) for w in s] 
                        + [get_idx_by_word(self.end_label)] for s in sentences]
        # target_onehot = self.get_onehot_wocab(target_idx)
        return target_idx

    def next_batch(self, batch_size, type):
        if type == 'train':
            file_id = 0
        if type == 'test':
            file_id = 1
        df = self.data[file_id].sample(batch_size ,replace=False)
        sentences = [df['question1'].values, df['question2'].values]
        sentences = [[self.clean_str(s).split() for s in q] for q in sentences]
        
        ecoder_input = self.get_encoder_input(sentences)
        decoder_input = self.get_decoder_input(sentences)
        target = self.get_target(sentences)

        return [encoder_input, decoder_input, target]

    # Original taken from https://github.com/facebookresearch/InferSent/blob/master/data.py
    def embed_batch(self, batch):
        max_len = np.max([len(x) for x in batch])
        embed = np.zeros((len(batch), max_len, 300))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[i, j, :] = self.word_vec[batch[i][j]]

        return t.from_numpy(embed).float()

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent.split():
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<p>'] = ''
        return word_dict
    
    def build_most_common_vocab(self, sentences):
        word_counts = collections.Counter(sentences)
        self.idx_to_word = [x[0] for x in word_counts.most_common(self.vocab_size - 2)] \
        + ['</s>'] + [self.unk_label]
        self.word_to_idx = {self.idx_to_word[i] : i for i in self.vocab_size}
    
    def sample_word_from_distribution(self, distribution):
        assert len(distribution) == self.vocab_size
        ix = np.random.choice(range(self.vocab_size), p=distribution.ravel())
        return self.idx_to_word[ix]
    
    def get_onehot_vocab(self, ids):
        batch_size = len(ids)
        max_seq_len = np.max([len(x) for x in ids])
        res = np.zeros((batch_size, max_seq_len, self.vocab_size),dtype=np.int32)
        for i in range(batch_size):
            for j in range(max_seq_len):
                if j < len(ids[i]):
                    res[i][j][ids[i][j]] = 1 
                else :
                    res[i][j][self.vocab_size - 1] = 1 # end symb
        return res

    def get_word_by_idx(self, idx):
        return self.idx_to_word[idx]

    def get_idx_by_word(self, w):
        if w in self.word_to_idx.keys():
            return self.word_to_idx[w]
        return self.word_to_idx[self.unk_label]

    def build_glove(self, word_dict):
        # create word_vec with glove vectors
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    self.word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(self.word_vec), len(word_dict)))

    def get_sentences_from_data(self):
        sentences = []
        for df in self.data:
            sentences += list(df['question1'].values) + list(df['question2'].values)
        return sentences

    def build_vocab(self, sentences):
        if sentences is None:
            sentences = self.get_sentences_from_data()
        sentences = [self.clean_str(s) for s in sentences]

        self.max_seq_len = np.max([len(s) for s in sentences]) + 1
        word_dict = self.get_word_dict(sentences)
        
        self.build_most_common_vocab(sentences)
        self.build_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))
