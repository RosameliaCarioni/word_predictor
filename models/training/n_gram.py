import math
import time
from tqdm import tqdm


class NGram:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.total_words = 0
        self.unique_words = 0

        # Occurrences count - [word][ids of words]
        self.unigram_count = {}
        self.trigram_count = {}
        self.bigram_count = {}

        # Probabilities  - [word][ids of words]
        self.unigram_prob = {}
        self.trigram_prob = {}
        self.bigram_prob = {}

        self.N = 3

    def process_files_split_train_test(self, files):
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as file:
                total_lines = sum(1 for line in file)  # Count the total lines first
                file.seek(0)  # Reset file pointer to the beginning
                i = 0
                max_i = int(0.85*total_lines)
                for line in tqdm(file, total=max_i, desc=f"Processing lines in {file_path}"):
                    if i <= max_i:
                        self.count_n_grams(line)
                        i += 1
                    else:
                        break
        self.probability_n_gram()

    def process_files(self, files):
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as file:
                total_lines = sum(1 for line in file)  # Count the total lines first
                file.seek(0)  # Reset file pointer to the beginning
                for line in tqdm(file, total=total_lines, desc=f"Processing lines in {file_path}"):
                    self.count_n_grams(line)
        self.probability_n_gram()

    def make_testset(self, files):
        test_file_path = 'ngram_test_set.txt'
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as file:
                total_lines = sum(1 for line in file)  # Count the total lines first
                i = 0
                max_i = int(0.85*total_lines)
                file.seek(0)  # Reset file pointer to the beginning
                for line in tqdm(file, total=total_lines, desc=f"Processing lines in {file_path}"):
                    if i > max_i:
                        with open(test_file_path, 'a') as test_file:
                            test_file.write(line)
                    i += 1

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
            previous_words = self.get_previous_words(words, i, self.N - 2)
            if len(previous_words) == self.N - 2:
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
            previous_words = self.get_previous_words(words, i, self.N - 1)
            if len(previous_words) == self.N - 1:
                if word in self.trigram_count:
                    if previous_words in self.trigram_count[word]:
                        self.trigram_count[word][previous_words] += 1
                    else:
                        self.trigram_count[word][previous_words] = 1
                else:
                    self.trigram_count[word] = {}
                    self.trigram_count[word][previous_words] = 1

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
                    file.write(f'{i} {word} {self.unigram_count[word]} \n')
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

                file.write(str(-1))  # end of file
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")


def main():
    trigram = NGram()
    files = ['/Users/ericbanzuzi/uni/KTH/NLP/word_predictor/data/clean_data/twitter.txt',
             '/Users/ericbanzuzi/uni/KTH/NLP/word_predictor/data/clean_data/articles.txt',
             #'/Users/ericbanzuzi/uni/KTH/NLP/word_predictor/data/clean_data/mobile_text.txt',
             '/Users/ericbanzuzi/uni/KTH/NLP/word_predictor/data/clean_data/news_summarization.txt']
    start = time.time()
    trigram.process_files_split_train_test(files)
    model_path = '../weights/ngram_model_small_final.txt'
    trigram.save_model(model_path)
    print(f"Training finished in {time.time() - start} seconds")
    # print('\nCreating test set...')
    # trigram.make_testset(files)
    # print('DONE!')


if __name__ == '__main__':
    main()
