import collections
import os
import re
import torch as t
from torch.autograd import Variable
import numpy as np
import pandas as pd

def clean_str(string):
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
    string = re.sub(r'\W+', ' ', string)
    string = string.lower()
    return string.strip()


class BatchLoader:
    def __init__(self, vocab_size=25000, sentences=None, path=''):
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

        self.df_from_file = None
        
        self.quora_data_files = [path + 'data/quora/train.csv', path + 'data/quora/test.csv']
        self.snli_path = '../InferSent/dataset/SNLI/'
        self.glove_path = '/home/aleksey.zotov/InferSent/dataset/GloVe/glove.840B.300d.txt'

        if sentences is None:
            self.read_train_test_dataset()

        self.build_vocab(sentences)

    def get_encoder_input(self, sentences):
        return [Variable(t.from_numpy(
            self.embed_batch([s + [self.end_label] for s in q]))).float() for q in sentences]
    
    def get_decoder_input(self, sentences): 
        enc_inp = self.embed_batch([s + [self.end_label] for s in sentences[0]]) 
        dec_inp = self.embed_batch([[self.go_label] + s for s in sentences[1]]) 
        return [Variable(t.from_numpy(enc_inp)).float(), Variable(t.from_numpy(dec_inp)).float()]
    
    def get_target(self, sentences):
        sentences = sentences[1]
        max_seq_len = np.max([len(s) for s in sentences]) + 1
        target_idx = [[self.get_idx_by_word(w) for w in s] 
                        + [self.get_idx_by_word(self.end_label)] * (max_seq_len - len(s))
                        for s in sentences] 
        # target_onehot = self.get_onehot_wocab(target_idx)
        return Variable(t.from_numpy(np.array(target_idx, dtype=np.int64))).long()

    def get_raw_input_from_sentences(self, sentences):
        sentences = [clean_str(s).split() for s in sentences] 
        return Variable(t.from_numpy(self.embed_batch(sentences))).float()

    def input_from_sentences(self, sentences):
        sentences = [[clean_str(s).split() for s in q] for q in sentences]
        
        encoder_input_source, encoder_input_target = self.get_encoder_input(sentences)
        decoder_input_source, decoder_input_target = self.get_decoder_input(sentences)
        target = self.get_target(sentences)

        return [encoder_input_source, encoder_input_target, 
                decoder_input_source, decoder_input_target,
                target]

    def next_batch(self, batch_size, type, return_sentences=False):
        if type == 'train':
            file_id = 0
        if type == 'test':
            file_id = 1
        df = self.data[file_id].sample(batch_size ,replace=False)
        sentences = [df['question1'].values, df['question2'].values]
        
        input = self.input_from_sentences(sentences)

        if return_sentences:
            return input, [[clean_str(s).split() for s in q] for q in sentences]
        else:
            return input

    def next_batch_from_file(self, batch_size, file_name, return_sentences=False):
        if self.df_from_file is None:
            self.df_from_file = pd.read_csv(file_name)
            self.cur_file_point = 0
        
        # file ends
        if self.cur_file_point == len(self.df_from_file):
            self.cur_file_point = 0
            return None

        end_point = min(self.cur_file_point + batch_size, len(self.df_from_file))
        df = self.df_from_file.iloc[self.cur_file_point:end_point]
        sentences = [df['question1'].values, df['question2'].values]
        self.cur_file_point = end_point


        input = self.input_from_sentences(sentences)

        if return_sentences:
            return input, [[clean_str(s).split() for s in q] for q in sentences]
        else:
            return input

    # Original taken from https://github.com/facebookresearch/InferSent/blob/master/data.py
    def embed_batch(self, batch):
        max_len = np.max([len(x) for x in batch])
        embed = np.zeros((len(batch), max_len, 300))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] == self.go_label or batch[i][j] == self.end_label:
                    continue
                if batch[i][j] in self.word_vec.keys(): 
                    embed[i, j, :] = self.word_vec[batch[i][j]]
                else:
                    embed[i, j, :] = self.word_vec['null']

        return embed

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
        + [self.unk_label] + ['</s>']
        self.word_to_idx = {self.idx_to_word[i] : i for i in range(self.vocab_size)}
    
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

    def build_input_vocab(self, sentences):
        word_dict = self.get_word_dict(sentences)
        self.build_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))

    def build_output_vocab(self, sentences):
        self.max_seq_len = np.max([len(s) for s in sentences]) + 1
        text = ' '.join(sentences).split()
        self.build_most_common_vocab(text)

    def build_vocab(self, sentences):
        if sentences is None:
            sentences = self.get_sentences_from_data()
        sentences = [clean_str(s) for s in sentences]

        self.build_input_vocab(sentences)
        self.build_output_vocab(sentences)
        
        
    # READ DATA 
    def read_train_test_dataset(self):
        quora = [pd.read_csv(f)[['question1', 'question2']] for f in self.quora_data_files]
        snli = self.get_nli()
        self.data = [q.append(s, ignore_index=True) for q,s in zip(quora,snli)]

    def get_nli(self):
        # https://github.com/facebookresearch/InferSent (c)
        data_path = self.snli_path
        
        s1 = {}
        s2 = {}
        target = {}

        dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}

        for data_type in ['train', 'dev', 'test']:
            s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
            s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
            s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
            target[data_type]['path'] = os.path.join(data_path,
                                                     'labels.' + data_type)

            s1[data_type]['sent'] = np.array([line.rstrip() for line in
                                     open(s1[data_type]['path'], 'r')])
            s2[data_type]['sent'] = np.array([line.rstrip() for line in
                                     open(s2[data_type]['path'], 'r')])
            target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                    for line in open(target[data_type]['path'], 'r')])

            assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
                len(target[data_type]['data'])

            print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                    data_type.upper(), len(s1[data_type]['sent']), data_type))

        train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
                 'label': target['train']['data']}

        tr1 = s1['train']['sent'][target['train']['data'] == 0] # entailment
        tr2 = s2['train']['sent'][target['train']['data'] == 0]
        train_df = pd.DataFrame(data=np.array([tr1, tr2]).T, columns=['question1', 'question2'])

        ts1 = s1['test']['sent'][target['test']['data'] == 0] # entailment
        ts2 = s2['test']['sent'][target['test']['data'] == 0]
        test_df = pd.DataFrame(data=np.array([ts1, ts2]).T, columns=['question1', 'question2'])

        return [train_df, test_df]
            
