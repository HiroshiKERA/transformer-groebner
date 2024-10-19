import torch
import torch.nn as nn
import math
from torch import Tensor, LongTensor, FloatTensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape, self.pe[:, :x.shape[1], :].shape)
        return self.pe[:, :x.shape[1], :]


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        positions = torch.arange(x.shape[1]).long().to(x.device)
        return self.embedding(positions)


class ContinuousEmbedding(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=128, model='ffn'):
        super().__init__()
        
        # self.embedding = nn.Sequential(nn.Linear(1, hidden_dim), 
        #                                 nn.ReLU(), 
        #                                 nn.Linear(hidden_dim, embedding_dim))
        if 'ffn' in model:
            num_hidden_layers = int(model[3:]) - 1 if len(model) > 3 else 0
            hidden_layers = []
            for _ in range(num_hidden_layers):
                hidden_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
            self.embedding = nn.Sequential(nn.Linear(1, hidden_dim), 
                                        nn.ReLU(), 
                                        *hidden_layers,
                                        nn.Linear(hidden_dim, embedding_dim))
            
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        return x


class HybridEmbedding(nn.Module):
    def __init__(self, discrete_vocab_size, continuous_hidden_dim=128, embedding_dim=128, padding_idx=-100, continuous_embedding_model='ffn'):
        super().__init__()
        self.embedding_dim              = embedding_dim
        self.d_embedding                = nn.Embedding(discrete_vocab_size, embedding_dim, padding_idx=padding_idx)
        self.continuous_embedding_model  = continuous_embedding_model
        
        self.c_embedding = ContinuousEmbedding(hidden_dim=continuous_hidden_dim, embedding_dim=embedding_dim, model=continuous_embedding_model)
    
    def forward(self, x: Tensor, x_contineuous_labels: Tensor = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
            labels: Tensor, shape [batch_size, continous_vocab_size], float for target tokens and NaN for others.
        """
        embedding = torch.zeros(*x.shape, self.embedding_dim).to(x.device)

        is_continuous = x_contineuous_labels.isfinite()
        
        discrete_ids = x[~is_continuous].long()
        embedding[~is_continuous, :] = self.d_embedding(discrete_ids)

        continuous_ids = x_contineuous_labels[is_continuous]
        embedding[is_continuous, :] = self.c_embedding(continuous_ids.view(-1)).to(embedding.dtype) # accumulate continuous embeddings (supposed exclusive though)
        
        return embedding
