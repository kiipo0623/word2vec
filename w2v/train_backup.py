import numpy as np
from backup import *

## (1) text load -> raw text
path = './data.txt'
data = load_data(path)

## (2) preprocess(text)
# input : raw text
# output : tokenized text, word2id dict, id2word dict
seq_len = 10
corpus, word2id, id2word = preprocess(data, seq_len)
# 문장 18개

## (3) co_matrix
vocab_size = len(word2id.keys())
window_size = 2
co_matrix = build_co_matrix(corpus, vocab_size, window_size)
# co_matrix.shape = (63, 63)

## corpus에서 target word와 context word 추출
context, target = split_context_target(corpus, window_size)

## (4) 각 context를 onehot vector(numpy) 로 바꿔주자
context_onehot = convert_to_onehot(context, vocab_size)
target_onehot = convert_to_onehot(target, vocab_size)
print('target - X')
# print(f'{type(target_onehot)}') # list
# print(f'{len(target_onehot)}') # 18
print(target_onehot.shape)
print('context - Y')
# print(type(context_onehot)) # list
# print(len(context_onehot)) # 18
print(context_onehot.shape) # 6, 4, 64
# 18개의 target list, 각 target vector.shape = (3, 63)
#                       context vector.shape = (3,4,63)


## (5) 모델 학습을 시켜보아요
# batch_size, vocab_size, hidden_size, seq_len, window_size
batch_size = 1
skipgram = SkipGram(1, vocab_size, 5, seq_len, window_size)
for sentenceX, sentenceY in zip(target_onehot, context_onehot):
    for x, y in zip(sentenceX, sentenceY):
        skipgram.forward(x, y)
# argment : vocab_size, hidden_size
# hidden layer : simple vers.이므로 하나
# input layer -> hidden layer -> output layer
# 각 layer 사이에서는 행렬곱만
# output 에 softmax를 취한게 최종 prediction y

## 멍청하지만 수식을 짜내보아요
## params : W_input(vocab_size, hidden_size), W_output(hidden_size, vocab_size)
## X : input, Y : predicted output

###### forward ######
# hidden = matmul(W_input, X)
# output = matmul(W_output, hidden)
# for c in contexts --> window_size*2 개만큼 돌아감
# 각 c에 대해서 softmax_c 를 적용
# loss = loss layer1(output, context1)  +  loss layer2(output, context2)
# loss layer:
#     prediction = softmax(output)
#     prediction이랑 context가 얼마나 다른가 ? 를 cross-entropy metric으로 구해서 반환
# Y = softmax(output)
'''cross entropy metric이란 ?
p = [ softmax(a[i]) for i in range(batch_size) ]
L = - sum( y[j]*log(p[j]) for j in range(batch_size))


L를 a[i]에 대해서 편미분하면
p[i] - y[i] 가 되는데, y는 정답 레이블만 1로 표시한 one-hot vector이므로
결국 정답 레이블에서만 1을 빼주고, 나머지 i에 대해서는 0을 빼주는 것이므로 의미 ㄴ
'''
## -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

## (6) loss를 구해보아요
# cross-entropy(y, t)
# skip-gram에서는 하나의 center word를 받아서 window_size*2 만큼의 (앞뒤) context word를 predict하기 때문에
# loss는 각 context word에서의 loss의 총합이 되어야 함

###### backward ######
'''
1. loss backward -> 각 context word에 대해서 구한 뒤 모두 더한게 loss lambda
y(prediction), t(answer)
batch_size
dx = y
dx[[0, 1, 2, ..., s], t] -= 1
-> numpy의 ndarray라서 대충 저렇게 하면 각 arange 행의, 그에 상응하는 t[i]열이 뽑힘

[[one_hot prediction for token 0],
 [one_hot prediction for token 1],
 [one_hot prediction for token 2],
 ...
 [one_hot prediction for token s]]


 2. hidden backward
 dW = gradient*x.T

'''
