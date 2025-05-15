import torch
import torch.nn as nn

class GatedMultimodalFusion(nn.Module):
    def __init__(self, mm_dim, clinical_dim):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(mm_dim + clinical_dim, 128),
            nn.ReLU(),
            nn.Linear(128, mm_dim),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(mm_dim + clinical_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

    def forward(self, mm_feat, clinical_feat):
        gate_input = torch.cat([mm_feat, clinical_feat], dim=-1)
        gate_weights = self.gate_network(gate_input)
        return self.fusion_layer(torch.cat([mm_feat * gate_weights, clinical_feat], dim=-1)), gate_weights
    