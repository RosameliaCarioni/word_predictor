import math
import codecs
import time
import heapq
import os


class NGram:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.total_words = 0
        self.unique_words = 0

        # Occurrences count - [word][ids of words]
        self.unigram_count = {}

        # Probabilities  - [word][ids of words]
        self.trigram_prob = {}
        self.bigram_prob = {}

        # Linear interpolation
        self.lambda_1 = 0.90
        self.lambda_2 = 0.09
        self.lambda_3 = 0.01 - 10e-6
        self.lambda_4 = 10e-6
        self.N = 3

        self.prob_cache = {}  # stores results for future

    def read_model(self, file_path):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: True if the entire file could be processed, False otherwise.
        """
        try:
            with codecs.open(file_path, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                # process the rest of the file
                for _ in range(self.unique_words):
                    data = f.readline().strip().split(' ')  # from [index, word, count]
                    self.word_to_id[data[1]] = int(data[0])
                    self.id_to_word[int(data[0])] = data[1]
                    self.unigram_count[int(data[0])] = int(data[2])

                # process rest of the file
                for line in f.readlines():
                    if line.strip() == '-1':  # end of the file
                        break
                    data = line.strip().split(' ')  # from [first word, prev words, log prob]
                    if len(data) == 4:
                        prev_words = [data[1].strip('(').replace(',', ''), data[2].strip(')')]
                        if int(data[0]) in self.trigram_prob:
                            self.trigram_prob[int(data[0])][(int(prev_words[0]), int(prev_words[1]))] = float(data[3])
                        else:
                            self.trigram_prob[int(data[0])] = {}
                            self.trigram_prob[int(data[0])][(int(prev_words[0]), int(prev_words[1]))] = float(data[3])
                    else:
                        if int(data[0]) in self.bigram_prob:
                            self.bigram_prob[int(data[0])][int(data[1])] = float(data[2])
                        else:
                            self.bigram_prob[int(data[0])] = {}
                            self.bigram_prob[int(data[0])][int(data[1])] = float(data[2])

                print('Successfully read the model')
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(file_path))
            return False

    def get_unigram_prob(self, word_id):
        if word_id in self.unigram_count:
            return self.unigram_count[word_id] / self.total_words
        else:
            return 0

    def get_bigram_prob(self, word_id, prev_word_id):
        if word_id in self.bigram_prob and prev_word_id in self.bigram_prob[word_id]:
            return math.exp(self.bigram_prob[word_id][prev_word_id])
        else:
            return 0

    def get_trigram_prob(self, word_id, prev_word_ids):
        if word_id in self.trigram_prob and prev_word_ids in self.trigram_prob[word_id]:
            return math.exp(self.trigram_prob[word_id][prev_word_ids])
        else:
            return 0

    def interpolate_prob(self, word_id, prev_word_ids):
        cache_key = (word_id, prev_word_ids)
        # check if the probability was already computed before
        if cache_key in self.prob_cache:
            return self.prob_cache[cache_key]

        unigram_prob = self.get_unigram_prob(word_id)
        bigram_prob = self.get_bigram_prob(word_id, prev_word_ids[-1]) if len(prev_word_ids) > 0 else 0
        trigram_prob = self.get_trigram_prob(word_id, prev_word_ids)

        prob = (self.lambda_1 * trigram_prob +
                self.lambda_2 * bigram_prob +
                self.lambda_3 * unigram_prob +
                self.lambda_4)

        self.prob_cache[cache_key] = prob
        return prob

    def predict_next_word(self, context, number_of_suggestions=5):
        if len(context) == 0:
            return []

        filter_words = False if context[-1] == ' ' else True
        current_word = None
        if filter_words:
            words_split = context.split()
            prev_word_ids = [self.word_to_id[word.lower()] if word.lower() in self.word_to_id else -1
                             for word in words_split][-3:-1]
            current_word = words_split[-1].lower()
        else:
            prev_word_ids = [self.word_to_id[word.lower()] if word.lower() in self.word_to_id else -1
                             for word in context.split()][-2:]

        heap = []
        for word_id in self.word_to_id.values():
            if filter_words:
                word = self.id_to_word[word_id]
                if not word.startswith(current_word):
                    continue

            prob = self.interpolate_prob(word_id, tuple(prev_word_ids))
            if len(heap) < number_of_suggestions:
                heapq.heappush(heap, (prob, word_id))
            else:
                heapq.heappushpop(heap, (prob, word_id))

        sorted_candidates = sorted(heap, key=lambda item: item[0], reverse=True)
        return [self.id_to_word[word_id] for prob, word_id in sorted_candidates]


def initialize_model(model_path='models/weights/ngram_model_small.txt'):
    model = NGram()
    if os.path.exists(model_path):
        model.read_model(model_path)
        print(f'Model {model_path} loaded.')
    else:
        print(f"File {model_path} does not exist.")
    return model


if __name__ == '__main__':
    file_path = 'weights/ngram_model_small.txt'
    trigram = initialize_model(file_path)
    start = time.time()
    trigram.read_model(file_path)
    print(f"Reading finished in {time.time() - start} seconds")
    print('Vocabulary size:', trigram.unique_words)
    start = time.time()
    print(trigram.predict_next_word("what is", 5))
    print(f"Prediction finished in {time.time() - start} seconds")
