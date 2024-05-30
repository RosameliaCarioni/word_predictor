import tkinter as tk
from tkinter import ttk
from models import rnn, n_gram, transformer
from models.training.rnn import run, RNN

class WordPredictorApp:
    def __init__(self, root, model, max_predictions=5):
        self.root = root
        self.model = model
        self.max_predictions = max_predictions
        self.saved_characters = 0

        self.root.title("Word Predictor")

        self.text_var = tk.StringVar()
        self.text_entry = ttk.Entry(root, textvariable=self.text_var, width=50)
        self.text_entry.grid(row=0, column=0, columnspan=max_predictions)

        self.prediction_buttons = []
        for i in range(max_predictions):
            btn = ttk.Button(root, text="", command=lambda i=i: self.insert_prediction(i))
            btn.grid(row=1, column=i)
            self.prediction_buttons.append(btn)

        self.saved_chars_label = ttk.Label(root, text="Saved Characters: 0")
        self.saved_chars_label.grid(row=2, column=0, columnspan=max_predictions)

        self.text_entry.bind("<KeyRelease>", self.update_predictions)

    def update_predictions(self, event):
        current_text = self.text_var.get()
        predictions = self.model.predict_next_word(current_text, self.max_predictions)
        for i in range(self.max_predictions):
            if i < len(predictions):
                self.prediction_buttons[i].config(text=predictions[i], state="normal")
            else:
                self.prediction_buttons[i].config(text="", state="disabled")

    def insert_prediction(self, index):
        current_text = self.text_var.get()
        words = current_text.split()
        if not words:
            return

        # Calculate saved characters
        last_word = words[-1]
        predicted_word = self.prediction_buttons[index].cget("text")
        self.saved_characters += len(predicted_word) - len(last_word)

        # Update text with prediction
        new_text = " ".join(words[:-1] + [predicted_word])
        self.text_var.set(new_text)

        # Update saved characters label
        self.saved_chars_label.config(text=f"Saved Characters: {self.saved_characters}")

        self.text_entry.focus_set()
        self.text_entry.icursor(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    rnn_model = rnn.initialize_rnn() #RNN() # maybe call main function that returns a RNN() class model rnn.initialize_rnn()
    app = WordPredictorApp(root, rnn_model, max_predictions=5)
    root.mainloop()