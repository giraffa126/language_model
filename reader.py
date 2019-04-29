# coding: utf8
import sys
import os
import collections
import random
import pickle

class DataReader(object):
    def __init__(self, vocab_path, data_path, vocab_size=1024, batch_size=64):
        """ init
        """
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        if not os.path.exists(vocab_path):
            self._word_to_id = self._build_vocab(data_path)
            with open(vocab_path, "w") as ofs:
                pickle.dump(self._word_to_id, ofs)
        else:
            with open(vocab_path, "r") as ifs:
                self._word_to_id = pickle.load(ifs)
        self._data = self._build_data(data_path, self._word_to_id)

    def _build_vocab(self, filename):
        with open(filename, "r") as ifs:
            data = ifs.read().replace("\n", " ").split()
        counter = collections.Counter(data)
        count_pairs = counter.most_common(self._vocab_size - 3)

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(3, len(words) + 3)))
        word_to_id["<pad>"] = 0
        word_to_id["<bos>"] = 1
        word_to_id["<eos>"] = 2
        print("vocab words num: ", len(word_to_id))
        return word_to_id

    def _build_data(self, filename, word_to_id, is_shuffle=True):
        with open(filename, "r") as ifs:
            lines = ifs.readlines()
            data = list(map(lambda x: x.strip().split(), lines))
            random.shuffle(data)
        data = list(map(lambda x: ["<bos>"] + x + ["<eos>"], data))
        data = list(map(lambda x: [word_to_id.get(w, word_to_id["<unk>"]) for w in x], data))
        return data

    def _padding_batch(self, batch):
        pading_batch = [[], []]
        batch_max_len = max([len(x) for x in batch])
        for line in batch:
            inputs = line + [self._word_to_id["<pad>"]] * (batch_max_len - len(line)) 
            outputs = line[1:-1] + [self._word_to_id["<pad>"]] * (batch_max_len - len(line)) 
            pading_batch[0].append(inputs)
            pading_batch[1].append(outputs)
        return pading_batch

    def get_vocab_size(self):
        return len(self._word_to_id)

    def batch_generator(self):
        curr_size = 0
        batch = []
        for line in self._data:
            curr_size += 1
            batch.append(line)
            if curr_size >= self._batch_size:
                yield self._padding_batch(batch)
                batch = []
                curr_size = 0
        if curr_size > 0:
            yield self._padding_batch(batch)

if __name__ == "__main__":
    reader = DataReader("data/vocab.pkl", "data/train.txt")
    for batch in reader.batch_generator():
        print(batch)
