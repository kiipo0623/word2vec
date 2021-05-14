import numpy as np
from word2vec import *
from train import *
import news
import pickle

# 하이퍼파라미터 설정
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 1

# 데이터 읽기
corpus, word_to_id, id_to_word = news.load_data()
vocab_size = len(word_to_id)
print(vocab_size)

contexts, target = split_context_target(corpus, window_size)
print("context0", contexts[0])
print("target0", target[0])
print("context1", contexts[1])
print("target1", target[1])
print("context2", contexts[2])
print("target2", target[2])
print("context3", contexts[3])
print("target3", target[3])
print("context4", contexts[4])
print("target4", target[4])


#모델 등 생성
model = CBOW_dev(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Train(model, optimizer)

#학습 시작
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

#나중에 사용할 수 있도록 필요한 데이터 저장
word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'news_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dumps(params, f, -1)