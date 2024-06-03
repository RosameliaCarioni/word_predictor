import torch
import os
import sys
from models.training.rnn import RNN
#from training.rnn import run, RNN
from transformers import AutoTokenizer
import nltk
from nltk.corpus import words

# Download the list of words if not already done
#nltk.download('words')
nltk.download('wordnet')


from torch import nn, optim

class RNNpredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def filter_vocab_by_prefix(self, vocab, prefix):
        if prefix == None:
            return vocab
        return {token: idx for token, idx in vocab.items() if token.startswith(prefix)}

    def mask_logits_by_vocab(self, logits, filtered_vocab):
        mask = torch.full_like(logits, float('-inf'))
        for token, idx in filtered_vocab.items():
            mask[idx] = logits[idx]
        return mask
    
    def remove_last_word(self, input_string, cut=True):
        last_space_index = input_string.rfind(' ')
        if last_space_index == -1:
            return None, input_string
        else:
            if cut:
                # if prompt is longer than seven words, cut it
                words = input_string.split()
                last_seven_words = words[-7:]
                result = ' '.join(last_seven_words)
                if input_string[-1] == " ":
                    result += " "
                last_space_index = result.rfind(' ')
                input_string = result
            return input_string[:last_space_index], input_string[last_space_index+1:]

    def predict_next_word(self, prompt, number_of_suggestions, max_subwords=5):
        self.model.eval()

        input_text = prompt
        vocab = self.tokenizer.get_vocab()
        hidden = None
        english_words = set(words.words())
        unused_tokens = [token for token in self.tokenizer.vocab if token.startswith('[unused')]

        # remove last word from prompt (word that is supposed to be predicted)
        prompt, prefix = self.remove_last_word(prompt, True)
        full_prompt, _ = self.remove_last_word(input_text, False)
        if prompt == None:
            tokens = [self.tokenizer.cls_token_id]
        else:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(tokens).unsqueeze(0).to(self.device)  # Add batch dimension
        
        next_words = []
        #print(prompt, prefix)
        first_pass = []

        for i in range(number_of_suggestions):
            generated_subwords = []
            for _ in range(max_subwords):
                with torch.no_grad():
                    outputs, hidden = self.model(input_ids, hidden)
                    next_token_logits = outputs.squeeze()  # Get the logits for the last token
                    
                if len(generated_subwords) == 0:
                    # filter by prefix
                    filtered_vocab = self.filter_vocab_by_prefix(vocab, prefix)
                    # Mask the logits based on the filtered vocabulary
                    masked_logits = self.mask_logits_by_vocab(next_token_logits, filtered_vocab)
                    # Normalize the masked logits to get probabilities
                    probs = torch.softmax(masked_logits, dim=-1)
                    if i == 0:
                        first_pass = probs.topk(number_of_suggestions).indices.tolist()
                    next_token_id = first_pass[i]
                else: 
                    next_token_id = next_token_logits.topk(number_of_suggestions).indices.tolist()[i]


                # Decode the generated subwords so far
                subword_text = self.tokenizer.decode([next_token_id], clean_up_tokenization_spaces=True)
                #print("subword", subword_text, subword_text.lower() in english_words)
                # Check if the last token completes a word
                if (not subword_text.startswith("##") and len(generated_subwords) > 0):  # Check if it's not a continuation of a word
                    break
                if (subword_text.lower() in english_words and len(generated_subwords) == 0):
                    generated_subwords.append(next_token_id)
                    break
                if subword_text == self.tokenizer.pad_token or subword_text in unused_tokens:
                    break
                if subword_text.startswith("##") and len(generated_subwords) == 0:
                    break

                generated_subwords.append(next_token_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(self.device)], dim=1).to(self.device)  # Append the predicted token to the input

            # Decode the generated subwords to form the next word
            #print(generated_subwords)
            next_word = self.tokenizer.decode(generated_subwords, clean_up_tokenization_spaces=True).strip()
            next_words.append(next_word)
        return next_words
    
    
def initialize_rnn():
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print( "Running on", device , "when initializing")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    try:
        model = torch.load('models/weights/gru_no_padding.model', map_location=device).to(device)
    except FileNotFoundError:
            print(f"File not found: models/weights/gru_no_padding.model")
    except Exception as e:
            print(f"An error occurred: {e}")


    return RNNpredictor(model, tokenizer, device)

    
if __name__ == '__main__':
    rnn = initialize_rnn()
    print(rnn.predict_next_word("on top of the w", 5))
    print(rnn.predict_next_word("My name is k", 5))
    print(rnn.predict_next_word("I am from Germ", 5))
    print(rnn.predict_next_word("Thi", 5))
    print(rnn.predict_next_word("", 5))