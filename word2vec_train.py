import numpy as np
from word2vec import *
from train import *

#설정
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

#load data
# path = './data.txt'
# data = load_data(path)
# print(data)
data = 'Wrote some songs about Ricky.'

#preprocess
#seq_len = 0 한 번에 들어갈 때 너무 짧으면 padding
#우선 data.txt에 있는 것 중 제일 긴 것보다 크게 해
corpus, word_to_id, id_to_word = preprocess(data)
print(corpus)
print(word_to_id)

#target word, context word
context, target = split_context_target(corpus, 1)
print(context[0])
print(target[0])

#convert to onehot
vocab_size = len(word_to_id.keys())
context_onehot = convert_to_onehot(context, vocab_size)
target_onehot = convert_to_onehot(target, vocab_size)
print("context", context_onehot[0])
print("target", target_onehot[0])

#여기까지는 [전체[문장[단어]]]
#이 밑으로는 [문장[단어]]
#우선 문장-단어로 하고 전체를 받는걸로 키우는 게 나을 듯..?

model = CBOW(vocab_size, hidden_size=5)
optimizer = SGD()
trainer = Train(model, optimizer)

trainer.fit(context_onehot, target_onehot, max_epoch, batch_size)
trainer.plot()