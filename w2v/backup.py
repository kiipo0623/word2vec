# util - functions
import numpy as np
from layer import MatMul, SoftmaxWithLoss

def load_data(path):
    data = open(path).readlines()
    '''
    [sentence1,
    sentence2,
    ..., sentence]
    '''

    ### customize - print ###
    print(f'{len(data)} data load from {path}')
    print(data[:5])
    #########################

    return data


def preprocess(corpus, seq_len):
    # input : raw text - sentence의 list다
    # output : tokenized text, word2id dict, id2word dict
    text = [text.strip().lower().replace('.', ' .').split() for text in corpus]
    text = [t + ['<pad>'] * (seq_len - len(t)) for t in text]
    word2id = {'<pad>': 0}
    id2word = {0: '<pad>'}
    for words in text:
        for word in words:
            if word not in word2id:
                new_id = len(word2id)
                word2id[word] = new_id
                id2word[new_id] = word

    corpus = [np.array([word2id[w] for w in words]) for words in text]

    return corpus, word2id, id2word


def build_co_matrix(corpus, vocab_size, W):
    # input : corpus, vocab_size, window_size
    # output : co_matrix

    co_matrix = np.zeros((vocab_size, vocab_size))
    # vocab size X vocab size
    for sentence in corpus:
        for index, word_id in enumerate(sentence):
            for i in range(1, W + 1):
                left_idx = index - i
                right_idx = index + i
                print(sentence)
                print(left_idx, right_idx)

                # left_idx = -2, right_idx = 2
                if left_idx >= 0:
                    left_word_id = sentence[left_idx]
                    # print(f'sentence : {sentence}')
                    # print(f'word_id : {word_id}')
                    # print(f'context word id: {left_word_id}')
                    co_matrix[word_id, left_word_id] += 1
                if right_idx < len(sentence):
                    # print(co_matrix)
                    right_word_id = sentence[right_idx]
                    co_matrix[word_id, right_word_id] += 1
                    # print(f'sentence : {sentence}')
                    # print(f'word_id : {word_id}')
                    # print(f'context word id: {right_word_id}')
    return co_matrix


def split_context_target(corpus, window_size):
    targets, contexts = list(), list()
    for sentence in corpus:
        _targets = sentence[window_size:-window_size]
        _contexts = []

        for i in range(window_size, len(sentence) - window_size):
            context = []
            for t in range(-window_size, window_size + 1):
                if t:
                    context.append(sentence[i + t])
            _contexts.append(context)
        targets.append(np.array(_targets))
        contexts.append(np.array(_contexts))

    return np.array(contexts), np.array(targets)


def convert_to_onehot(corpus, vocab_size):
    ## (4) 각 context를 onehot vector(numpy) 로 바꿔주자
    # context vector든 center vector든 one-hot 형태로 받으므로 전처리단에서 얘가 필요
    # output : numpy

    onehots = []
    for sentence in corpus:
        Nrow = sentence.shape[0]
        if sentence.ndim == 1:
            one_hot = np.zeros((Nrow, vocab_size))
            for idx, word_id in enumerate(sentence):
                one_hot[idx, word_id] = 1

        # context word
        if sentence.ndim == 2:
            Ncol = sentence.shape[1]  # window size
            one_hot = np.zeros((Nrow, Ncol, vocab_size))
            for idx_1, word_ids in enumerate(sentence):
                for idx_2, word_id in enumerate(word_ids):
                    one_hot[idx_1, idx_2, word_id] = 1

        onehots.append(one_hot)

    # corpus 전체를 그냥 .. 한번에 one hot 때리고자 할때..
    # one_hot = np.zeros(vocab_size)
    # for idx, word_id in enumerate(corpus):
    #     one_hot[idx, word_id] = 1

    return np.array(onehots)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid(x):
    ## sigmoid
    return 1 / (1 + np.exp(-x))

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
