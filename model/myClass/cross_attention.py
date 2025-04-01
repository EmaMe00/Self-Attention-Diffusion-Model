from package import *

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # Scaling per stabilità numerica

        # Proiezioni lineari per Query, Key e Value
        self.to_q = nn.Linear(dim, dim, bias=False)  
        self.to_k = nn.Linear(dim, dim, bias=False)  
        self.to_v = nn.Linear(dim, dim, bias=False)  

        # Output della cross-attention
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.Dropout(dropout)  # Dropout per regolarizzazione
        )

    def forward(self, x, cond):
        """
        x: Feature map dell'input principale (B, N, C)
        cond: Feature map dell'input condizionale (B, M, C)
        """
        B, N, C = x.shape
        _, M, _ = cond.shape
        H = self.num_heads

        # Proiezione e reshaping per multi-head attention
        q = self.to_q(x).reshape(B, N, H, C // H).permute(0, 2, 1, 3)  # (B, H, N, C/H)
        k = self.to_k(cond).reshape(B, M, H, C // H).permute(0, 2, 1, 3)  # (B, H, M, C/H)
        v = self.to_v(cond).reshape(B, M, H, C // H).permute(0, 2, 1, 3)  # (B, H, M, C/H)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, M)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)

        return self.to_out(out)