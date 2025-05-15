import torch
import torch.nn as nn
from .backbone import USFeatureExtractor, ClinicalEncoder
from .attention import CrossModalAttention
from .fusion import GatedMultimodalFusion

class OvarianSurvivalModel(nn.Module):
    def __init__(self, us_dim=512, wsi_dim=768, clinical_dim=256):
        super().__init__()
        self.us_extractor = USFeatureExtractor(output_dim=us_dim)
        self.clinical_encoder = ClinicalEncoder(hidden_dim=clinical_dim)
        self.cross_attn = CrossModalAttention(us_dim, wsi_dim)
        self.gated_fusion = GatedMultimodalFusion(self.cross_attn.hidden_dim, clinical_dim)
        self.risk_head = nn.Linear(256, 1)

    def forward(self, batch):
        us_feat = self.us_extractor(batch['us'])
        cross_feat = self.cross_attn(us_feat, batch['wsi'])
        clinical_feat = self.clinical_encoder(batch['clinical'])
        fused_feat, gate_weights = self.gated_fusion(cross_feat, clinical_feat)
        return self.risk_head(fused_feat).squeeze(-1), gate_weights
    