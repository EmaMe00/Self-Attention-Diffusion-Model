from myClass.package import *
from model_param import *

class SwinEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rearrange = Rearrange('b c h w -> b (h w) c')
        
    def forward(self, x):
        x = self.rearrange(x).contiguous()
        return x
    
class ShiftedWindowMSA(nn.Module):
    def __init__(self, device, emb_size, num_heads, window_size=7, shifted=True):
        super().__init__()
        self.device = device
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.linear1 = nn.Linear(emb_size, 3*emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.pos_embeddings = nn.Parameter(torch.randn(window_size*2 - 1, window_size*2 - 1))
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

    def forward(self, x):
        h_dim = self.emb_size / self.num_heads
        height = width = int(np.sqrt(x.shape[1]))
        x = self.linear1(x).contiguous()
        
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size).contiguous()
        
        if self.shifted:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1, 2)).contiguous()
        
        x = rearrange(
            x, 
            'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', 
            w1=self.window_size, w2=self.window_size, H=self.num_heads
        ).contiguous()
        
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1).contiguous(), K.squeeze(-1).contiguous(), V.squeeze(-1).contiguous()
        wei = (Q @ K.transpose(4, 5)).contiguous() / np.sqrt(h_dim)
        
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding.contiguous()
        
        if self.shifted:
            row_mask = torch.zeros((self.window_size**2, self.window_size**2), device=self.device)
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size).contiguous()
            wei[:, :, -1, :] += row_mask.contiguous()
            wei[:, :, :, -1] += column_mask
        
        wei = F.softmax(wei, dim=-1).contiguous() @ V.contiguous()
        
        x = rearrange(
            wei, 
            'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)', 
            w1=self.window_size, w2=self.window_size, H=self.num_heads
        ).contiguous()
        x = rearrange(x, 'b h w c -> b (h w) c').contiguous()
        
        return self.linear2(x).contiguous()
    
class MLP(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4*emb_size),
            nn.GELU(),
            nn.Linear(4*emb_size, emb_size),
        )
    
    def forward(self, x):
        return self.ff(x).contiguous()
    
class SwinEncoder(nn.Module):
    def __init__(self, device, emb_size, num_heads, window_size=7):
        super().__init__()
        self.device = device
        self.WMSA = ShiftedWindowMSA(self.device, emb_size, num_heads, window_size, shifted=False)
        self.SWMSA = ShiftedWindowMSA(self.device, emb_size, num_heads, window_size, shifted=True)
        self.ln = nn.LayerNorm(emb_size)
        self.MLP = MLP(emb_size)
        
    def forward(self, x):
        x = x + self.WMSA(self.ln(x).contiguous()).contiguous()
        x = x + self.MLP(self.ln(x).contiguous()).contiguous()
        x = x + self.SWMSA(self.ln(x).contiguous()).contiguous()
        x = x + self.MLP(self.ln(x).contiguous()).contiguous()
        return x

def pad_or_resize_image(x, window_size):
    _, _, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0).contiguous()

class SingleSwinBlock(nn.Module):
    def __init__(self, device, emb_size=96, in_channels=3, num_heads=4, window_size=8):
        super(SingleSwinBlock, self).__init__()
        self.device = device
        self.embedding = SwinEmbedding()
        self.swin_encoder = SwinEncoder(self.device, emb_size, num_heads, window_size)
        self.output_projection = nn.Linear(emb_size, in_channels)
    
    def forward(self, x):
        B, _, H, W = x.shape[0], x.shape[1], *x.shape[2:]
        x = self.embedding(x).contiguous()
        x = self.swin_encoder(x).contiguous()
        x = self.output_projection(x).contiguous()
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        return x
