# util - functions
import numpy as np


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


###### model layers ######
### center word를 받아서 그에 대한 context vector * window_size 를 predict --> loss는 한 번에
###

class SkipGram():
    def __init__(self, batch_size, vocab_size, hidden_size, seq_len, window_size, lr=0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.window_size = window_size
        self.batch_size = batch_size
        self.lr = lr  # learning_rate

        self.hidden_layer = linear(vocab_size, hidden_size)
        self.output_layer = linear(hidden_size, vocab_size)
        self.loss = cross_entropy_loss()

    def forward(self, X, Y):
        '''
        x : 64차원 단어 벡터 하나
        Y : (각 문장 단어 수 6, 각 context word 수 4, 64차원)
        (vocab_size, hidden_size)의 hidden layer에 넣고
        그 결과를 다시
        (hidden_size, sentence len)에 넣기
        '''
        h = self.hidden_layer.forward(X)
        o = self.output_layer.forward(h)
        loss = 0
        for i in range(self.window_size*2):
            loss += self.loss.forward(o, Y[i, :]) #(p, y) p:prediction y:target
        print(loss)
        return loss

    def backward(self, X, Y):
        pass


class linear():
    def __init__(self, I, O):
        ### I : input dim ###
        ### O : output dim ### ########################################################################################
        ## 레퍼런스에서는 파라미터 초깃값을 충분히 작게 해주기 위하여 0.01 을 곱해줌 ##
        self.W = np.random.randn(I, O).astype('f')
        self.grad = np.zeros((I, O))  # gradient of W
        ########################################################################################

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W)

    def backward(self, dout):
        ## forward : return x.dot.W
        ## compute gradient of X and W

        self.grad = np.dot(self.x.T, dout)  # gradient of W
        return np.dot(dout, self.W.T)  # gradient of X


class cross_entropy_loss():
    def __init__(self):
        # Prediction, Y_label을 받음
        self.params, self.grads = [], []
        self.prediction = None
        self.target = None

    def forward(self, p, y):
        prediction = softmax(p)
        self.prediction = prediction
        self.batch = p.shape[0]
        self.target = y # ndarray, (64,)
        print('target')
        print(type(y))
        print(y.shape)

        return -np.sum(np.log(prediction[np.arange(self.batch), y] + 1e-8)) / self.batch

    def backward(self):
        dout = self.prediction.copy()
        dout[np.arange(self.batch), self.target] -= 1
        print('dout[np.arange(self.batch), self.target] -= 1 확인용')

        ##################################
        # 이 이후에 gradient를 batch_size로 나눠주는데 그건 왜그런지 모르겟다.. 걍 취사 선택해도 될것 ㄱ같아서 스킵 #

        return dout
