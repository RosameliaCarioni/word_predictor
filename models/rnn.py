
class RNN:
    def __init__(self):
        pass

    def predict_next_word(self, words):
        return [f"{words}_{i}" for i in range(10)]