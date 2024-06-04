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

import nltk
from nltk.corpus import words

class RNNpredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def filter_vocab_by_prefix(self, vocab, prefix):
        if prefix == None:
            return vocab
        # return {token: idx for token, idx in vocab.items() if token.startswith(prefix)}
        filtered_vocab = {}
        filtered_subtokens = {}
        for token, idx in vocab.items():
            if token.startswith(prefix):
                filtered_vocab[token] = idx
            elif token.startswith('##') and token[2:].isalpha():
                filtered_subtokens[token] = idx
        return filtered_vocab, filtered_subtokens

    def mask_logits_by_vocab(self, logits, filtered_vocab):
        mask = torch.full_like(logits, float('-inf'))
        for token, idx in filtered_vocab.items():
            mask[idx] = logits[idx]
        return mask

    def mask_logits_by_subword(self, mask, logits, filtered_subwords):
        for token, idx in filtered_subwords.items():
            mask[idx] = logits[idx]
        return mask

    def remove_last_word(self, input_string, cut=True):
        last_space_index = input_string.rfind(' ')
        if last_space_index == -1:
            return None, input_string.lower()
        else:
            if cut:
                # if prompt is longer than seven words, cut it
                words = input_string.lower().split()
                last_seven_words = words[-7:]
                result = ' '.join(last_seven_words)
                if input_string[-1] == " ":
                    result += " "
                last_space_index = result.rfind(' ')
                input_string = result
            return input_string[:last_space_index], input_string[last_space_index + 1:]

    def predict_next_word(self, prompt, number_of_suggestions, max_subwords=3):
        self.model.eval()

        input_text = prompt
        hidden = None
        vocab = self.tokenizer.get_vocab()
        #nltk.download('words')
        english_words = set(words.words())

        # remove last word from prompt (word that is supposed to be predicted)
        prompt, prefix = self.remove_last_word(prompt, True)
        full_prompt, _ = self.remove_last_word(input_text, False)
        if prompt == None:
            tokens = [self.tokenizer.cls_token_id]
        else:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids_start = torch.tensor(tokens).unsqueeze(0).to(self.device)  # Add batch dimension
        input_ids = input_ids_start


        first_pass = []
        suggestions = []
        i = 0
        while len(suggestions) < number_of_suggestions:
            generated_subwords = []
            for subword in range(max_subwords):
                if len(suggestions) == 0 and i == 0:
                    with torch.no_grad():
                        outputs, hidden = self.model(input_ids, hidden)
                        next_token_logits = outputs.squeeze()  # Get the logits for the last token

                    # filter by prefix
                    filtered_vocab, filtered_subwords = self.filter_vocab_by_prefix(vocab, prefix)
                    # Mask the logits based on the filtered vocabulary
                    masked_logits = self.mask_logits_by_vocab(next_token_logits, filtered_vocab)
                    # Mask the logits for most common '##'
                    masked_logits = self.mask_logits_by_subword(masked_logits, next_token_logits, filtered_subwords)
                    # Normalize the masked logits to get probabilities
                    probs = torch.softmax(masked_logits, dim=-1)
                    first_pass = probs.topk(len(filtered_vocab)+len(filtered_subwords)).indices.tolist()
                    next_token_id = first_pass[i]
                elif len(generated_subwords) == 0:
                    try:
                        next_token_id = first_pass[i]
                    except IndexError:
                        return suggestions
                else:
                    # filter by prefix
                    filtered_vocab, _ = self.filter_vocab_by_prefix(vocab, generated_subwords[-1])
                    if len(filtered_vocab) == 0:
                        break

                    with torch.no_grad():
                        outputs, hidden = self.model(input_ids, hidden)
                        next_token_logits = outputs.squeeze()  # Get the logits for the last toke

                    # Mask the logits based on the filtered vocabulary
                    masked_logits = self.mask_logits_by_vocab(next_token_logits, filtered_vocab)
                    # Find most likely end
                    next_token_id = masked_logits.topk(1).indices.tolist()[0]

                # Decode the generated subwords so far
                subword_text = self.tokenizer.decode([next_token_id], clean_up_tokenization_spaces=True)
                # print("subword", subword_text, subword_text.lower() in english_words)

                # Check if the last token can complete a word
                if not subword_text.startswith('[unused') and subword_text != self.tokenizer.pad_token:
                    if subword == 0:
                        i += 1
                    # is the word complete?
                    if subword_text.lower() in english_words and len(generated_subwords) == 0:
                        suggestions.append(subword_text)
                        break
                    # Check if it's not a continuation of a word
                    if not subword_text.startswith("##") and len(generated_subwords) > 0:
                        break
                    if subword_text.startswith("##"):
                        # is the word complete?
                        if len(generated_subwords) == 0 and prefix + subword_text[2:] in english_words:
                            if prefix + subword_text[2:] not in suggestions:
                                suggestions.append(prefix + subword_text[2:])
                            break
                        else:
                            if len(generated_subwords) > 0:
                                if generated_subwords[-1] + subword_text[2:] in english_words:
                                    if generated_subwords[-1] + subword_text[2:] not in suggestions:
                                        suggestions.append(generated_subwords[-1] + subword_text[2:])
                                    break
                                else:
                                    generated_subwords.append(generated_subwords[-1] + subword_text[2:])
                                    next_token_id_input = self.tokenizer.encode(generated_subwords[-1] + subword_text[2:], add_special_tokens=False)
                                    input_ids = torch.cat([input_ids, torch.tensor([next_token_id_input]).to(self.device)], dim=1).to(self.device)  # Append the predicted token to the input
                            else:
                                generated_subwords.append(prefix + subword_text[2:])
                                next_token_id_input = self.tokenizer.encode(prefix + subword_text[2:], add_special_tokens=False)
                                input_ids = torch.cat([input_ids, torch.tensor([next_token_id_input]).to(self.device)], dim=1).to(self.device)  # Append the predicted token to the input
            input_ids = input_ids_start
        return suggestions
    
    
def initialize_rnn():
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print( "Running on", device , "when initializing")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    try:
        model = torch.load('models/weights/gru_with_padding.model', map_location=device).to(device)
    except FileNotFoundError:
            print(f"File not found: models/weights/gru_with_padding.model")
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