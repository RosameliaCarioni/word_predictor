import torch
class RNN:
    def __init__(self, model):
        self.model = model
        pass

    def predict_next_word(self, words, number_of_suggestions):
        return [f"{words}_{i}" for i in range(number_of_suggestions)]
    
def initialize_rnn():

    # if there is not a rnn model file with values, call train/rnn.py main function
    # then read model 
    # return RNN model that has a .predict_next_word() function
    pass

def predict_next_word(model, tokenizer, words, number_of_suggestions, max_subwords=10):
        return [f"{words}_{i}" for i in range(number_of_suggestions)]

def predict_next_word(model, tokenizer, prompt, max_subwords=5):
    model.eval()
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    hidden = None

    generated_subwords = []

    for _ in range(max_subwords):
        with torch.no_grad():
            outputs, hidden = model(input_ids, hidden)
            next_token_logits = outputs[:, -1, :]  # Get the logits for the last token
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()  # Get the ID of the highest probability token
        
        # Decode the generated subwords so far
        subword_text = tokenizer.decode([next_token_id], clean_up_tokenization_spaces=True)
        
        # Check if the last token completes a word
        if not subword_text.startswith("##"):  # Check if it's not a continuation of a word
            break

        generated_subwords.append(next_token_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)  # Append the predicted token to the input

    # Decode the generated subwords to form the next word
    next_word = tokenizer.decode(generated_subwords, clean_up_tokenization_spaces=True).strip()
    return next_word
    
if __name__ == '__main__':
    initialize_rnn()