from myClass.package import *

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=16, dropout=0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # Scaling per stabilità numerica

        # Proiezioni lineari per Query, Key e Value (uguali perché self-attention)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)  
        
        # Output della self-attention
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.Dropout(dropout)  # Dropout per regolarizzazione
        )

    def forward(self, x):
        """
        x: Feature map dell'input principale (B, N, C)
        """
        B, N, C = x.shape
        H = self.num_heads

        # Proiezione combinata e split in Q, K, V
        qkv = self.to_qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C/H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, C/H) per ciascuno

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)

        return self.to_out(out)