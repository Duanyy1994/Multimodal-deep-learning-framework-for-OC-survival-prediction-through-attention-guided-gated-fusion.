import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, us_dim, wsi_dim):
        super().__init__()
        self.hidden_dim = max(us_dim, wsi_dim)
        self.us_proj = nn.Linear(us_dim, self.hidden_dim) if us_dim != self.hidden_dim else nn.Identity()
        self.wsi_proj = nn.Linear(wsi_dim, self.hidden_dim) if wsi_dim != self.hidden_dim else nn.Identity()
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, us_feat, wsi_feat):
        us_proj = self.us_proj(us_feat)
        wsi_proj = self.wsi_proj(wsi_feat)
        q = self.query(us_proj).unsqueeze(1)
        k = self.key(wsi_proj).unsqueeze(2)
        attn_weights = self.softmax(torch.matmul(q, k))
        return attn_weights * us_proj + (1 - attn_weights) * wsi_proj
    