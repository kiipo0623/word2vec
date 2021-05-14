import pickle
import numpy as np
import os

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = 'news.train.npy'
vocab_file = 'news.vocab.pkl' #npy 파일 : numpy 배열을 파일의 형태로 저장

def load_vocab():
    vocab_path = dataset_dir + '/' + vocab_file

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
            return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    file_name = 'news1.txt'
    file_path = dataset_dir + '/' + file_name

    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word

def load_data():
    save_path = dataset_dir + '/' + save_file

    word_to_id, id_to_word = load_vocab()
    if os.path.exists(save_path):
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word

    file_name = 'news1.txt'
    file_path = dataset_dir + '/' + file_name

    words = open(file_path).read().replace('\n', '<eos>').strip().split()
    corpus = np.array([word_to_id[w] for w in words])

    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word