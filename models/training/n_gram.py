import math
import time


class NGram:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.total_words = 0
        self.unique_words = 0

        # Occurrences count - [word][ids of words]
        self.unigram_count = {}
        self.fourgram_count = {}
        self.trigram_count = {}
        self.bigram_count = {}

        # Probabilities  - [word][ids of words]
        self.unigram_prob = {}
        self.fourgram_prob = {}
        self.trigram_prob = {}
        self.bigram_prob = {}

        self.N = 4

    def process_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # self.count_unigrams(line)
                self.count_n_grams(line)
        self.probability_unigram()
        self.probability_n_gram()

    def count_unigrams(self, line):
        words = line.strip().split()
        for word in words:
            if word in self.word_to_id:
                self.unigram_count[word] += 1
            else:
                self.unigram_count[word] = 1
                self.word_to_id[word] = self.unique_words
                self.id_to_word[self.unique_words] = word
                self.unique_words += 1
            self.total_words += 1

    def get_previous_words(self, sentence, i, number_of_previous_words_considered):
        previous_words = []
        left_bound = max(0, i - number_of_previous_words_considered)

        for j in range(left_bound, i):
            previous_words.append(self.word_to_id[sentence[j]])
        return tuple(previous_words)

    def count_n_grams(self, line):
        words = line.strip().split()
        for i, word in enumerate(words):
            # UNIGRAM
            if word in self.word_to_id:
                self.unigram_count[word] += 1
            else:
                self.unigram_count[word] = 1
                self.word_to_id[word] = self.unique_words
                self.id_to_word[self.unique_words] = word
                self.unique_words += 1
            self.total_words += 1

            # BIGRAM
            previous_words = self.get_previous_words(words, i, self.N - 3)
            if len(previous_words) == self.N - 3:
                previous_word = previous_words[0]  # only one preceding word
                if word in self.bigram_count:
                    if previous_word in self.bigram_count[word]:
                        self.bigram_count[word][previous_word] += 1
                    else:
                        self.bigram_count[word][previous_word] = 1
                else:
                    self.bigram_count[word] = {}
                    self.bigram_count[word][previous_word] = 1

            # TRIGRAM
            previous_words = self.get_previous_words(words, i, self.N - 2)
            if len(previous_words) == self.N - 2:
                if word in self.trigram_count:
                    if previous_words in self.trigram_count[word]:
                        self.trigram_count[word][previous_words] += 1
                    else:
                        self.trigram_count[word][previous_words] = 1
                else:
                    self.trigram_count[word] = {}
                    self.trigram_count[word][previous_words] = 1

            # FOUR-GRAM 
            previous_words = self.get_previous_words(words, i, self.N - 1)
            if len(previous_words) == self.N - 1:
                if word in self.fourgram_count:
                    if previous_words in self.fourgram_count[word]:
                        self.fourgram_count[word][previous_words] += 1
                    else:
                        self.fourgram_count[word][previous_words] = 1
                else:
                    self.fourgram_count[word] = {}
                    self.fourgram_count[word][previous_words] = 1

    def probability_unigram(self):
        for word in self.unigram_count.keys():
            self.unigram_prob[word] = math.log(self.unigram_count[word] / self.total_words)

    def probability_n_gram(self):
        for word in self.bigram_count.keys():
            prev_words = self.bigram_count[word]
            self.bigram_prob[word] = {}
            for prev_word in prev_words.keys():
                # p (word|prev_word) =  c(word|prev_word)/ c(prev_word)
                self.bigram_prob[word][prev_word] = math.log(self.bigram_count[word][prev_word] /
                                                             self.unigram_count[self.id_to_word[prev_word]])

        for word in self.trigram_count.keys():
            prev_words = self.trigram_count[word]
            self.trigram_prob[word] = {}
            for prev_word in prev_words:
                self.trigram_prob[word][prev_word] = math.log(self.trigram_count[word][prev_word] /
                                                              self.bigram_count[self.id_to_word[prev_word[1]]][prev_word[0]])

        for word in self.fourgram_count.keys():
            prev_words = self.fourgram_count[word]
            self.fourgram_prob[word] = {}
            for prev_word in prev_words:
                self.fourgram_prob[word][prev_word] = math.log(self.fourgram_count[word][prev_word] /
                                                               self.trigram_count[self.id_to_word[prev_word[-1]]][prev_word[:-1]])

    def save_model(self, file_path):
        """
        Creates a list of rows to print of the language model.
        """
        try:
            with open(file_path, 'w') as file:
                # size of vocabulary and number of tokens
                file.write(f'{self.unique_words} {self.total_words} \n')

                # write vocabulary
                for i, word in self.id_to_word.items():
                    file.write(f'{i} {word} {self.unigram_prob[word]} \n')
                print('Successfully wrote unigram prob')

                # write bigram probabilities
                for word, prev_words in self.bigram_prob.items():
                    id_prev_word, prob_value = prev_words.popitem()
                    file.write(f'{self.word_to_id[word]} {id_prev_word} {prob_value} \n')
                print('Successfully wrote bigram prob')

                # write trigram probabilities
                for word, prev_words in self.trigram_prob.items():
                    id_prev_word, prob_value = prev_words.popitem()
                    file.write(f'{self.word_to_id[word]} {id_prev_word} {prob_value} \n')
                print('Successfully wrote trigram prob')

                # write fourgram probabilities
                for word, prev_words in self.fourgram_prob.items():
                    id_prev_word, prob_value = prev_words.popitem()
                    file.write(f'{self.word_to_id[word]} {id_prev_word} {prob_value} \n')
                print('Successfully wrote fourgram prob')

                file.write(str(-1))  # end of file
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")


def main():
    four_gram = NGram()
    # file_path = '/Users/rosameliacarioni/University/MSc/1_year/4_period/language engineering/word_predictor/data/clean_data/articles.txt'
    file_path = '/Users/ericbanzuzi/uni/KTH/NLP/word_predictor/data/twitter/en_US.twitter.txt'
    start = time.time()
    four_gram.process_file(file_path)
    file_path = './testing.txt'
    four_gram.save_model(file_path)
    print(f"Training finished in {time.time() - start} seconds")


if __name__ == '__main__':
    main()
