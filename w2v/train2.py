import numpy as np
from word2vec import *
from train import *
import ptb
import pickle


#하이퍼파라미터 설정
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

#데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = split_context_target(corpus, window_size)

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
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
