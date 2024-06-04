
from torch import nn, optim
class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) with optional GRU or LSTM units.

    Attributes:
    - no_of_output_symbols (int): Size of the output vocabulary.
    - embedding_size (int): Dimensionality of the embeddings.
    - hidden_size (int): Number of features in the hidden state.
    - num_layers (int): Number of recurrent layers.
    - use_GRU (bool): If True, use GRU; otherwise, use LSTM.
    - dropout (float): Dropout probability.
    - device (torch.device): Device for the model ('cpu', 'mps' or 'cuda').
    """
    def __init__(self, embedding_size, hidden_size, no_of_output_symbols, device, num_layers, use_GRU, dropout):
        super().__init__()
        self.no_of_output_symbols = no_of_output_symbols
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_GRU = use_GRU
        self.dropout = dropout

        # initialize layers
        self.embedding = nn.Embedding(no_of_output_symbols, embedding_size)
        if use_GRU == True:
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
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
        x_emb = self.embedding(x) # x_emb shape: (batch_size, max_seq_length, emb_dim)
        if self.use_GRU:
            output, hidden = self.rnn(x_emb, hidden) # output shape: (batch_size, max_seq_length, hidden)
        else:
            output, (h_n, c_n) = self.rnn(x_emb, hidden)  # LSTM expects a tuple (hidden state, cell state)
            hidden = (h_n, c_n)
            
        return self.output(output[:, -1, :]), hidden # logit shape: (batch_size, 1, vocab_size)
    
 