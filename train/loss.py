import torch

def cox_gate_loss(risk_scores, event_times, event_indicators, gate_weights, lambda_reg=0.01):
    sorted_idx = torch.argsort(-event_times)
    risk_scores = risk_scores[sorted_idx]
    event_indicators = event_indicators[sorted_idx]
    log_cumsum_hazard = torch.logcumsumexp(risk_scores, dim=0)
    cox_loss = -torch.mean((risk_scores - log_cumsum_hazard) * event_indicators)
    gate_reg = lambda_reg * torch.mean(torch.abs(gate_weights))
    return cox_loss + gate_reg
    