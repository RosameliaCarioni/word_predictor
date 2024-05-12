import math 

class NGram:
    def __init__(self):
       self.word_to_id = {}
       self.id_to_word = {} 
       self.total_words = 0
       self.unique_words = 0
       
       # Occurances count - [word][ids of words]
       self.unigram_count = {} 
       self.fourgram_count = {}
       self.trigram_count = {}
       self.bigram_count = {}

       # Probabilities  - [word][ids of words]
       self.unigram_prob = {}
       self.fourgram_prob = {}
       self.trigram_prob = {}
       self.bigram_prob = {}

       # Linear interpolation -> TODO: find values 
       self.lambda_1 = 0.4
       self.lambda_2 = 0.3
       self.lambda_3 = 0.2
       self.lambda_4 = 0.1 - 10e-6
       self.lambda_5 = 10e-6


       self.N = 4


    def read_model(self, file_path):
        """
        Creates a list of rows to print of the language model.
        """
        try:
            with open(file_path, 'w') as file:
                # size of vocabulary and number of tokens
                file.write(f'{self.unique_words} {self.total_words} \n')

                # write vocabulary
                for i, word in self.id_to_word.items():
                    file.write(f'{i} {word} {self.unigram_prob[word]}')
                print('Succesfully wrote unigram prob')

                # write bigram probabilities
                for word, prev_words in self.bigram_prob.items():
                   file.write(f'{self.word_to_id[word]} {prev_words} {self.bigram_prob[word][prev_words]}')
                print('Succesfully wrote bigram prob')

                # write trigram probabilities
                for word, prev_words in self.trigram_prob.items():
                   file.write(f'{self.word_to_id[word]} {prev_words} {self.trigram_prob[word][prev_words]}')
                print('Succesfully wrote trigram prob')

                # write fourgram probabilities
                for word, prev_words in self.fourgram_prob.items():
                   file.write(f'{self.word_to_id[word]} {prev_words} {self.fourgram_prob[word][prev_words]}')
                print('Succesfully wrote fourgram prob')

        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")

    def predict_next_word(self, words):
        
        return [f"{words}_{i}" for i in range(10)]
