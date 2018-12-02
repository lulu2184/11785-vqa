import pickle


class WordDict(object):
    def __init__(self, word_to_idx=None, idx_to_word=None):
        if word_to_idx is None:
            word_to_idx = {}
        if idx_to_word is None:
            idx_to_word = []
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

    def __len__(self):
        return len(self.idx_to_word)

    def tokenize(self, sentence):
        words = sentence.lower().replace(
            ',', '').replace(
            '?', '').replace(
            '\'s', ' \'s').split()
        tokens = []
        for w in words:
            if w not in self.word_to_idx:
                self.idx_to_word.append(w)
                self.word_to_idx[w] = len(self.idx_to_word) - 1
            tokens.append(w)
        return tokens

    def dump_to_file(self, filepath):
        pickle.dump([self.word_to_idx, self.idx_to_word], open(filepath, 'wb'))

    @classmethod
    def load_from_file(cls, filepath):
        word_to_idx, idx_to_word = pickle.load(open(filepath, 'rb'))
        dict = cls(word_to_idx, idx_to_word)
        return dict
