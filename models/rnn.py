
class RNN:
    def __init__(self):
        pass

    def predict_next_word(self, words, number_of_suggestions):
        return [f"{words}_{i}" for i in range(number_of_suggestions)]