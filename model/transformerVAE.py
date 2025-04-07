import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings
from model.building_blocks import PositionalEmbedding, Embedding, TransformerBlock, DecoderBlock
warnings.simplefilter("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=2, n_heads=8, z_dim=2):
        super(TransformerEncoder, self).__init__()
        
        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        self.mean_layer = nn.Linear(embed_dim, z_dim)
        self.logvar_layer = nn.Linear(embed_dim, z_dim)

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out,out,out)

        mean, logvar = self.mean_layer(out), self.logvar_layer(out)

        return mean, logvar
    
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_layers=6, expansion_factor=2, n_heads=8):
        super(TransformerDecoder, self).__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads) 
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Args:
            x: input vector from target
        Returns:
            out: output vector
        """
        for layer in self.layers:
            x = layer(x, x, x)    
        return x

class TransformerVAE(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, seq_length,num_layers=4, expansion_factor=2, n_heads=4, z_dim= 2):
        super(TransformerVAE, self).__init__()
        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        """

        self.lin_proj = nn.Linear(z_dim, embed_dim)
        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, z_dim=z_dim)
        self.decoder = TransformerDecoder(embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.lstm = nn.LSTM(embed_dim, embed_dim)
        self.linear_proj = nn.Linear(embed_dim, src_vocab_size)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """

        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        decoder_input = self.lin_proj(z)
        decode = self.decoder(decoder_input)
        out_lstm, _ = self.lstm(decode)
        out = self.linear_proj(out_lstm)
        return out, mean, logvar
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z
    
    def decode(self, z):
        decoder_input = self.lin_proj(z)
        decode = self.decoder(decoder_input)
        out_lstm, _ = self.lstm(decode)
        out = self.linear_proj(out_lstm)
        out = self.out_act(out)
        return out

