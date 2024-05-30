import torch
import os
import sys
from models.training.rnn import run, RNN
#from training.rnn import run, RNN
from transformers import AutoTokenizer
import nltk
from nltk.corpus import words

# Download the list of words if not already done
#nltk.download('words')
nltk.download('wordnet')


from torch import nn, optim

# Add the project root to the PYTHONPATH
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# class RNN(nn.Module):
#     """
#     There are two possible ways to write this class; either it tries to predict 
#     a whole word that consists of several tokens or it only predicts the next token
#     after a fixed (or variable) amount of input tokens; 
#     Another choice is whether to use a hidden state or not as an input to the forward pass
#     Or do a encoder - decoder structure?

#     I read somewhere that it is good to ... 
#     """
#     def __init__(self, embedding_size, hidden_size, no_of_output_symbols, device, num_layers, GRU):
#         super().__init__()
#         self.no_of_output_symbols = no_of_output_symbols
#         self.embedding_size = embedding_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.GRU = GRU

#         # initialize layers
#         self.embedding = nn.Embedding(no_of_output_symbols, embedding_size)
#         if GRU == True:
#             self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
#         else:
#             self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
#         self.output = nn.Linear( hidden_size, no_of_output_symbols )
#         self.device = device
#         self.to(device)

#     def forward(self, x, hidden):
#         """
#         x is a list of lists of size (batch_size, max_seq_length)
#         Each inner list contains word IDs and represents one datapoint (n words).
       
#         Returns:
#         the output from the RNN: logits for the predicted next word, hidden state
#         """
#         x_emb = self.embedding(x) # x_emb shape: (batch_size, max_seq_length, emb_dim)
#         output, hidden = self.rnn(x_emb, hidden) # output shape: (batch_size, max_seq_length, hidden)
        
#         return self.output(output[:, -1, :]), hidden # logit shape: (batch_size, 1, vocab_size)
    
 

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
        print(prompt, prefix)
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
                print("subword", subword_text, subword_text.lower() in english_words)
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
            print(generated_subwords)
            next_word = self.tokenizer.decode(generated_subwords, clean_up_tokenization_spaces=True).strip()
            #next_words.append((next_word, str(input_text) + str(next_word)))
            if full_prompt == None:
                next_words.append(str(next_word))
            else:
                next_words.append(str(full_prompt) + " " + str(next_word))

        return next_words
    
    def predict_next_word_tkinter(self, prompt, number_of_suggestions, max_subwords=5):
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
        print(prompt, prefix)
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
                print("subword", subword_text, subword_text.lower() in english_words)
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
            print(generated_subwords)
            next_word = self.tokenizer.decode(generated_subwords, clean_up_tokenization_spaces=True).strip()
            #next_words.append((next_word, str(input_text) + str(next_word)))
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

    if not os.path.exists('models/lstm.model'):  # checking if there is a file with this name
        model = run()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = torch.load('models/lstm.model', map_location=device).to(device)

    return RNNpredictor(model, tokenizer, device)

    # if there is not a rnn model file with values, call train/rnn.py main function
    # then read model 
    # return RNN model that has a .predict_next_word() function

    
if __name__ == '__main__':
    rnn = initialize_rnn()
    print(rnn.predict_next_word("on top of the w", 5))
    print(rnn.predict_next_word("My name is k", 5))
    print(rnn.predict_next_word("I am from Germ", 5))
    print(rnn.predict_next_word("Thi", 5))
    print(rnn.predict_next_word("", 5))