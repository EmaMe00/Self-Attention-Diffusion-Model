from myClass.package import *
from myClass.self_attention import *

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class DownBlock(nn.Module):
    def __init__(self, device, in_channels, out_channels, t_emb_dim, cond_emb_dim,
                 down_sample=True, num_layers=1, apply_attention_list=False, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.apply_attention_list = apply_attention_list
        self.device = device

        self.resnet_conv_first = nn.ModuleList([
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])

        self.resnet_conv_second = nn.ModuleList([
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
        ])

        self.self_attn = nn.ModuleList([
            SelfAttention(out_channels) if self.apply_attention_list else nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
        ])

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb):
        out = x

        for i in range(self.num_layers):
            
            # Resnet block of Unet
            resnet_input = out
            #print("PRIMA DOWN")
            #print(out.shape)
            #print(context.shape)
            out = self.resnet_conv_first[i](out)
               
            if isinstance(self.self_attn[i], SelfAttention):
                #print("APPLICO CROSS DOWN")
                B, C, H, W = out.shape
                out_flat = out.view(B, C, H * W).permute(0, 2, 1).to(self.device)  # (B, N, C)
                out_flat = self.self_attn[i](out_flat)
                out = out_flat.permute(0, 2, 1).view(B, C, H, W).to(self.device)
                out = nn.GroupNorm(8, C).to(self.device)(out)
                out = nn.SiLU()(out)
                #print("FINE CROSS DOWN")

            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    def __init__(self, device, in_channels, out_channels, t_emb_dim, cond_emb_dim, num_layers=1, apply_attention_list=False, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.apply_attention_list = apply_attention_list

        self.device = device

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )

        self.self_attn = nn.ModuleList([
            SelfAttention(out_channels) if self.apply_attention_list else nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers + 1)
        ])

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers+1)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )
    
    def forward(self, x, t_emb):
        out = x

        # First resnet block
        
        resnet_input = out
        #print("PRIMA MID")
        #print(out.shape)
        #print(context.shape)
        out = self.resnet_conv_first[0](out)
        #print("DOPO MID")
        #print(out.shape)
        #print(context.shape)

        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):

            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i+1](out)

            if isinstance(self.self_attn[i], SelfAttention):
                #print("APPLICO CROSS DOWN")
                B, C, H, W = out.shape
                out_flat = out.view(B, C, H * W).permute(0, 2, 1).to(self.device)  # (B, N, C)
                out_flat = self.self_attn[i](out_flat)
                out = out_flat.permute(0, 2, 1).view(B, C, H, W).to(self.device)
                out = nn.GroupNorm(8, C).to(self.device)(out)
                out = nn.SiLU()(out)
                #print("FINE CROSS DOWN")

            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)
        
        return out

class UpBlock(nn.Module):
    def __init__(self, device, in_channels, out_channels, t_emb_dim, cond_emb_dim , up_sample=True, num_layers=1, apply_attention_list=False, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.apply_attention_list = apply_attention_list
        self.up_sample = up_sample

        self.device = device

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )

        self.self_attn = nn.ModuleList([
            SelfAttention(out_channels) if self.apply_attention_list else nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        #print(f"x: {x.shape}")
        #print(f"out_down: {out_down.shape}")
        x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            #print("PRIMA UP")
            #print(out.shape)
            #print(context.shape)
            #print("DOPO UP")
            out = self.resnet_conv_first[i](out)
            #print(out.shape)
            #print(context.shape)

            if isinstance(self.self_attn[i], SelfAttention):
                #print("APPLICO CROSS DOWN")
                B, C, H, W = out.shape
                out_flat = out.view(B, C, H * W).permute(0, 2, 1).to(self.device)  # (B, N, C)
                out_flat = self.self_attn[i](out_flat)
                out = out_flat.permute(0, 2, 1).view(B, C, H, W).to(self.device)
                out = nn.GroupNorm(8, C).to(self.device)(out)
                out = nn.SiLU()(out)
                #print("FINE CROSS DOWN")

            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

        return out

class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    def __init__(self, model_config, device):
        super().__init__()
        self.im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.apply_attention_list_down = model_config['apply_attention_down']
        self.apply_attention_list_mid = model_config['apply_attention_mid']
        self.apply_attention_list_up = model_config['apply_attention_up']
        self.dropout = model_config['dropout']

        self.device = device
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(self.im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.device, self.down_channels[i], self.down_channels[i+1], self.t_emb_dim, cond_emb_dim=self.im_channels, 
                                        down_sample=self.down_sample[i], num_layers=self.num_down_layers, apply_attention_list=self.apply_attention_list_down[i], dropout=self.dropout, ))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.device, self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim, cond_emb_dim=self.im_channels,
                                      num_layers=self.num_mid_layers, apply_attention_list=self.apply_attention_list_mid[i], dropout=self.dropout, ))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.device, self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, cond_emb_dim=self.im_channels, up_sample=self.down_sample[i], num_layers=self.num_up_layers, apply_attention_list=self.apply_attention_list_up[i], dropout=self.dropout, ))
        
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, self.im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W
        
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)

        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
            
        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out
