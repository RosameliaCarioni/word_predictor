import torch
import os
import training.rnn_parallel
from transformers import AutoTokenizer
class RNN:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer#
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
    
    def remove_last_word(self, input_string):
        last_space_index = input_string.rfind(' ')
        if last_space_index == -1:
            return input_string, None
        else:
            return input_string[:last_space_index], input_string[last_space_index+1:]

    def predict_next_word(self, prompt, number_of_suggestions, max_subwords=5):
        self.model.eval()

        vocab = self.tokenizer.get_vocab()
        hidden = None

        # remove last word from prompt (word that is supposed to be predicted)
        # TODO: should we also cut the prompt to be of a certain length (no longer than e.g. 10 words)?
        prompt, prefix = self.remove_last_word(prompt)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(tokens).unsqueeze(0).to(self.device)  # Add batch dimension
        
        next_words = []

        for i in range(number_of_suggestions):
            generated_subwords = []
            for _ in range(max_subwords):
                with torch.no_grad():
                    outputs, hidden = self.model(input_ids, hidden)
                    next_token_logits = outputs.squeeze()  # Get the logits for the last token
                    next_token_id = torch.argmax(next_token_logits, dim=-1).item()  # Get the ID of the highest probability token
                    next_token_ids = next_token_logits.topk(5).indices.tolist()

                if len(generated_subwords) == 0:
                    # filter by prefix
                    filtered_vocab = self.filter_vocab_by_prefix(vocab, prefix)
                    # Mask the logits based on the filtered vocabulary
                    masked_logits = self.mask_logits_by_vocab(next_token_logits, filtered_vocab)
                    # Normalize the masked logits to get probabilities
                    probs = torch.softmax(masked_logits, dim=-1)
                    next_token_id = probs.topk(5).indices.tolist()[i]
                else: 
                    next_token_id = next_token_logits.topk(5).indices.tolist()[i]


                # Decode the generated subwords so far
                subword_text = self.tokenizer.decode([next_token_id], clean_up_tokenization_spaces=True)
                # Check if the last token completes a word
                if not subword_text.startswith("##") and len(generated_subwords) > 0:  # Check if it's not a continuation of a word
                    break

                generated_subwords.append(next_token_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(self.device)], dim=1).to(self.device)  # Append the predicted token to the input

            # Decode the generated subwords to form the next word
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
    print( "Running on", device )

    if not os.path.exists('model.pth'):  # checking if there is a file with this name
        model = training.rnn_parallel.run()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = torch.load('model.pth')

    return RNN(model, tokenizer, device)
    # if there is not a rnn model file with values, call train/rnn.py main function
    # then read model 
    # return RNN model that has a .predict_next_word() function

    
if __name__ == '__main__':
    initialize_rnn()