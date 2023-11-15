import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BiLSTM_Model(nn.Module):
    def __init__(self, embedding):
        super(BiLSTM_Model, self).__init__()
        self.hidden_size = 128  # used in later procedure
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)    
        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hidden_size, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.5)  
        self.fc = nn.Linear(in_features = self.hidden_size*2, out_features=3)

    def forward(self, input):
        out = self.embedding(input)
        # LSTM layer
        lstm_out, _ = self.lstm(out)
        # Concatenate the forward and backward hidden states
        lstm_out = torch.cat((lstm_out[:, -1, :self.hidden_size], lstm_out[:, 0, self.hidden_size:]), dim=1)
        # Fully connected layer
        out = self.fc(lstm_out)
        
        return out
    
#AE-BiLSTM模型
class AEBiLSTM_Model(nn.Module):
    def __init__(self, embedding):
        super(AEBiLSTM_Model, self).__init__()
        self.hidden_size = 128  # used in later procedure
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)    
        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hidden_size, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.5)  # set num_layers 2 and bidirectional True to function as BiLSTM
        self.fc = nn.Linear(in_features = self.hidden_size*2, out_features=3)
    
    def forward(self, input, aspect):
        input = self.embedding(input)
        batch_size, seq_len, emb_ = input.size()
        aspect = self.embedding(aspect)
        aspect = aspect.unsqueeze(1).expand(-1, seq_len, -1)
        
        input = torch.cat((input, aspect), dim = 1)
        # LSTM layer
        lstm_out, _ = self.lstm(input)
        # Concatenate the forward and backward hidden states
        lstm_out = torch.cat((lstm_out[:, -1, :self.hidden_size], lstm_out[:, 0, self.hidden_size:]), dim=1)
        # Fully connected layer
        out = self.fc(lstm_out)
        
        return out

#AE-BiLSTM模型+soft attention
class AEBiLSTMWithSoftAttention_Model(nn.Module):
    def __init__(self, embedding):
        super(AEBiLSTMWithSoftAttention_Model, self).__init__()
        self.hidden_size = 128
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hidden_size, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.attention_dim = 256
        self.W_a = nn.Linear(in_features=self.hidden_size*2, out_features=self.attention_dim)
        self.W_b = nn.Linear(in_features=300, out_features=self.attention_dim) # corresponding to the size of input
        self.v = nn.Linear(in_features=self.attention_dim, out_features=1)
        self.fc = nn.Linear(in_features=self.hidden_size*2, out_features=3)

    def forward(self, input, aspect):
        input_embed = self.embedding(input)
        batch_size, seq_len, emb_dim = input_embed.size()

        aspect_embed = self.embedding(aspect)
        aspect_embed = aspect_embed.unsqueeze(1).expand(-1, seq_len, -1)

        combined_input = torch.cat((input_embed, aspect_embed), dim=1)

        # LSTM layer
        lstm_out, _ = self.lstm(combined_input)

        # Attention calculation
        attention_scores = self.v(torch.tanh(self.W_a(lstm_out) + self.W_b(combined_input)))
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_output = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected layer
        out = self.fc(attention_output)

        return out

#AE-BiLSTM模型+self attention
class AEBiLSTMWithSelfAttention_Model(nn.Module):
    def __init__(self, embedding):
        super(AEBiLSTMWithSelfAttention_Model, self).__init__()
        self.hidden_size = 128
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hidden_size, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.attention_dim = 256
        self.W_q = nn.Linear(in_features=self.hidden_size*2, out_features=self.attention_dim)
        self.W_k = nn.Linear(in_features=self.hidden_size*2, out_features=self.attention_dim)
        self.W_v = nn.Linear(in_features=self.hidden_size*2, out_features=self.attention_dim)
        self.fc = nn.Linear(in_features=self.attention_dim, out_features=3)

    def forward(self, input, aspect):
        input_embed = self.embedding(input)
        batch_size, seq_len, emb_dim = input_embed.size()

        aspect_embed = self.embedding(aspect)
        aspect_embed = aspect_embed.unsqueeze(1).expand(-1, seq_len, -1)

        combined_input = torch.cat((input_embed, aspect_embed), dim=1)

        # LSTM layer
        lstm_out, _ = self.lstm(combined_input)

        # Attention calculation (scaled dot-product attention)
        q = self.W_q(lstm_out)
        k = self.W_k(lstm_out)
        v = self.W_v(lstm_out)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Fully connected layer
        out = self.fc(attention_output[:, -1, :])

        return out
