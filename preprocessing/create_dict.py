import os
import json
import numpy as np

from word_dictionary import WordDict

EMBEDDING_DIMENSION = 300


def create_dict(data_dir):
    word_dict = WordDict()
    questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]

    for path in files:
        question_path = os.path.join(data_dir, path)
        questions = json.load(open(question_path))['questions']
        for question in questions:
            word_dict.tokenize(question['question'])
    return word_dict


def create_glove_embedding(idx_to_word, glove_file):
    word_to_embedding = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    embedding_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is {}'.format(embedding_dim))
    weights = np.zeros((len(idx_to_word), embedding_dim), dtype=np.float)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = map(float, vals[1:])
        word_to_embedding[word] = np.array(vals)
    for idx, word in enumerate(idx_to_word):
        if word in word_to_embedding:
            weights[idx] = word_to_embedding[word]
    return weights, word_to_embedding


if __name__ == '__main__':
    word_dict = create_dict('data')
    embedding_dim = EMBEDDING_DIMENSION
    glove_file = 'data/glove/glove.6B.{}.txt'.format(embedding_dim)
    weights, word_to_embedding = create_glove_embedding(
        word_dict.idx_to_word, glove_file)
    np.save('data/glove6b_init_{}.npy'.format(embedding_dim), weights)
