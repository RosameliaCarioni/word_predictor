import codecs
from datetime import datetime
import json
from pathlib import Path
import os
import glob
import numpy as np
import torch 
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer
from tqdm import tqdm



class WPDataset(Dataset):
    """
    A class loading clean from txt files to be used as an input 
    to PyTorch DataLoader.

    Datapoints are sequences of words (tokenized) + label (next token). If the 
    words have not been seen before (i.e, they are not found in the 
    'word_to_id' dict), they will be mapped to the unknown word '<UNK>'.
    """

    def __init__(self, filenames, tokenizer, n):
        self.sequences = []
        self.labels = []
        for filename in filenames:
            try :
                # Read the datafile
                with codecs.open(filename, 'r', 'utf-8') as f:
                    lines = f.read().split('\n')
                    for line in lines :
                        line_tokens = tokenizer.tokenize(line_tokens)
                        k = 0
                        while k < len(line_tokens)-n:
                            for i in range(1, n+1):
                                self.sequences.append([c for c in line_tokens[k:i+k]+[0]*(n-i)])
                                self.labels.append(line_tokens[i+k])
                            k += n
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"An error occurred: {e}")
                    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class RNN(nn.Module):
    """
    There are two possible ways to write this class; either it tries to predict 
    a whole word that consists of several tokens or it only predicts the next token
    after a fixed (or variable) amount of input tokens; 
    Another choice is whether to use a hidden state or not as an input to the forward pass
    Or do a encoder - decoder structure?

    I read somewhere that it is good to ... 
    """
    def __init__(self, embedding_size, hidden_size, no_of_output_symbols, device):
        super().__init__()
        self.no_of_output_symbols = no_of_output_symbols
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # initialize layers
        self.embedding = nn.Embedding(no_of_output_symbols, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        #self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear( hidden_size, no_of_output_symbols )
        self.device = device
        self.to(device)

    def forward(self, x, hidden):
        """
        x is a list of lists of size (batch_size, max_seq_length)
        Each inner list contains word IDs and represents one datapoint (n words).
       
        Returns:
        the output from the RNN: logits for the predicted next word, hidden state
        """

        x_tensor = torch.tensor(x).to(self.device)
        x_emb = self.embedding(x_tensor) # x_emb shape: (batch_size, max_seq_length, emb_dim)
        output, hidden = self.rnn(x_emb, hidden) # output shape: (batch_size, max_seq_length, hidden)
        
        return self.output(output[:, -1, :]), hidden # logit shape: (batch_size, 1, vocab_size)
    
    
def pad_sequence(batch, pad_symbol): #=tokenizer.pad_token):
    """
    Applies padding if the number of tokens in sequences differs within one batch.
    Only applies padding to the sequence, not the label.
    """
    seq, label = zip(*batch)
    max_seq_len = max(map(len, seq))
    max_label_len = max(map(len, label))
    padded_seq = [[b[i] if i < len(b) else pad_symbol for i in range(max_seq_len)] for b in seq]
    padded_label = [[l[i] if i < len(l) else pad_symbol for i in range(max_label_len)] for l in label]
    return padded_seq, padded_label


def evaluate(dataloader, rnn_model):
    correct, incorrect = 0,0
    for seq, label in dataloader:
        prediction, _ = rnn_model(seq)
        _, predicted_tensor = prediction.topk(1)
        
        if label == predicted_tensor:
            correct += 1
        else:
            incorrect += 1

    print( "Correctly predicted words    : ", correct )
    print( "Incorrectly predicted words  : ", incorrect )


def train():

    # ================ Hyper-parameters ================ #
    
    batch_size = 64
    embedding_size = 16
    hidden_size = 25
    seq_length = 5      # number of tokens used as a datapoint
    learning_rate = 0.001
    epochs = 30
    

    # ====================== Data ===================== #

    # select files with text for training (will also be used for test and validation dataset)
    mod_path = Path(__file__).parent.absolute()
    directory = os.path.join(mod_path, "data/clean_data/")
    txt_files = glob.glob(os.path.join(directory, '*.txt'))
    txt_files = [os.path.basename(file) for file in txt_files]
    txt_files = ["/Users/kathideckenbach/Documents/Machine Learning Master/Year 1/P4/Language Engineering/Assignments/word_predictor/data/clean_data/twitter.txt"]

    # choose tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # set up dataloaders
    dataset = WPDataset(filenames=txt_files, tokenizer=tokenizer, n=seq_length)

    # split the dataset into train, validation and test set
    size_dataset = len(dataset)
    datapoints = list(range(size_dataset))
    np.random.shuffle(datapoints)
    train_split = int(0.8 * size_dataset)
    val_split = int(0.05 * size_dataset) + train_split
    training_indices = datapoints[:train_split]
    validation_indices = datapoints[train_split:val_split]
    test_indices =  datapoints[val_split:]

    # create a dataloader for each of the sets
    training_sampler = SubsetRandomSampler(training_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)


    # ==================== Training ==================== #
    # Reproducibility
    np.random.seed(5719)

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print( "Running on", device )
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=training_sampler) # colate_fn = pad_sequence
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    print( "There are", len(dataset), "datapoints and ", tokenizer.vocab_size, "unique tokens in the dataset" ) 
    

    # set up model
    rnn_model = RNN(embedding_size, hidden_size, no_of_output_symbols=tokenizer.vocab_size, device=device).to(device)

    optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    rnn_model.train()

    for epoch in range(epochs):
        total_loss = 0
        hidden = None
        with tqdm(train_dataloader, desc="Epoch {}".format(epoch + 1)) as tepoch:
            for sequence, label in tepoch:
                sequence, label = sequence.to(device), label.to(device)
                optimizer.zero_grad()
                logits, hidden = rnn_model(sequence, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())  # Detach hidden states to avoid backprop through the entire sequence
                    
                loss = criterion(logits.squeeze(), torch.tensor(label).to(device))
                loss.backward()
                
                #clip_grad_norm_(rnn_model.parameters(), 5)
                optimizer.step()
                total_loss += loss
        print("Epoch", epoch, "loss:", total_loss.detach().item() )
        total_loss = 0

        if epoch % 10 == 0:
            print("Evaluating on the validation data...")
            evaluate(val_dataloader, rnn_model)

    # ==================== Save the model  ==================== #

    dt = str(datetime.now()).replace(' ','_').replace(':','_').replace('.','_')
    newdir = 'model_' + dt
    os.mkdir( newdir )
    torch.save( rnn_model.state_dict(), os.path.join(newdir, 'rnn.model') )

    settings = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'embedding_size': embedding_size
    }
    with open( os.path.join(newdir, 'settings.json'), 'w' ) as f:
        json.dump(settings, f)

    # ==================== Evaluation ==================== #

    rnn_model.eval()
    print( "Evaluating on the test data..." )

    print( "Number of test sentences: ", len(test_dataloader) )
    print()

    evaluate(test_dataloader, rnn_model)

    
if __name__ == '__main__':
    train()

