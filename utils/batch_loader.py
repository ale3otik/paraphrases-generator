import collections
import os
import re
import torch
import numpy as np
import pandas as pd

class BatchLoader:
    def __init__(self, sentences=None, path='../../'):
        '''
            Build vocab for sentences or for data files in path if None. 
        '''
        self.data_files = [path + 'data/quora/train.txt',
                           path + 'data/quora/test.txt']
        self.glove_path = [path + 'data/glove.840B.300d.txt']

        self.blind_symbol = ''
        self.pad_token = '_'
        self.go_token = '>'
        self.end_token = '|'

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

    def next_batch(self, batch_size, type):
        if type == 'train':
            file_id = 0
        if type == 'test':
            file_id = 1
        df = self.data[file_id].sample(batch_size ,replace=False)
        sentences = [df['question1'].values, df['question2'].values]
        embed = [self.embed_batch(batch) for batch in sentences]
        return embed
        
    # Original taken from https://github.com/facebookresearch/InferSent/blob/master/data.py
    def embed_batch(self, batch):
        batch = [self.clean_str(s) for s in batch]
        lengths = np.max([len(x) for x in batch])
        embed = np.zeros((max_len, len(batch), 300))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.from_numpy(embed).float()

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


    def build_glove(self, word_dict):
        # create word_vec with glove vectors
        self.word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    self.word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(self.word_vec), len(word_dict)))

    def get_sentences_from_data(self):
        for df in self.data:
            sentences += list(df['question1'].values) + list(df['question2'].values)
        return sentences

    def build_vocab(self, sentences):
        if sentences is None:
            sentences = self.get_sentences_from_data()
        sentences = [self.clean_str(s) for s in sentences]
        word_dict = self.get_word_dict(sentences)
        build_glove(self, word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))
