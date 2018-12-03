import json
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_LENGTH = 14


def create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def load_dataset(data_dir, name, img_id_to_val):
    question_path = os.path.join(
        data_dir, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(data_dir, 'cache', '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    entries = []
    for question, answer in zip(questions, answers):
        img_id = question['image_id']
        entries.append(create_entry(img_id_to_val[img_id], question, answer))

    return entries


class VQADataset(Dataset):
    def __init__(self, name, dictionary, data_dir='data'):
        super(VQADataset, self).__init__()
        assert name in ['train', 'val']

        answer_to_label_path = os.path.join(data_dir, 'cache',
                                            'trainval_answer_to_label.pkl')
        label_to_answer_path = os.path.join(data_dir, 'cache',
                                            'trainval_label_to_answer.pkl')
        self.answer_to_label = pickle.load(open(answer_to_label_path), 'rb')
        self.label_to_answer = pickle.load(open(label_to_answer_path), 'rb')
        self.answer_candidates_number = len(self.answer_to_label)
        self.dictionary = dictionary
        self.image_id_to_idx = pickle.load(
            open(os.path.join(data_dir, '{}36_imgid2idx.pkl'.format(name))))

        h5_path = os.path.join(data_dir, '{}36.hdf5'.format(name))
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))

        self.entries = load_dataset(data_dir, name, self.image_id_to_idx)

        # This will add q_token in each entry of the dataset.
        # -1 represent nil, and should be treated as padding_idx in embedding
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:MAX_LENGTH]
            if len(tokens) < MAX_LENGTH:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (
                        MAX_LENGTH - len(tokens))
                tokens = padding + tokens
            entry['q_token'] = tokens

        self.features = torch.from_numpy(self.features)

        # convert to tensor object
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

        # self.v_dim = self.features.size(2)

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.answer_candidates_number)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, question, target

    def __len__(self):
        return len(self.entries)
