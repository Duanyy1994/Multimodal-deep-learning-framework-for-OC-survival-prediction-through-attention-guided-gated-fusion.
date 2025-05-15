import torch
from data.dataset import build_dataloader
from model.survival_model import OvarianSurvivalModel
from train.trainer import Trainer
from train.loss import cox_gate_loss

DATA_CFG = {
    'clinical_path': './data/clinical.csv',
    'wsi_dir': './data/wsi_features',
    'us_dir': './data/ultrasound_images'
}

MODEL_CFG = {
    'us_dim': 512,
    'wsi_dim': 768,
    'clinical_dim': 256
}

TRAIN_CFG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 5e-5,
    'lambda_reg': 0.01,
    'early_stop_patience': 15
}

if __name__ == '__main__':
    train_loader = build_dataloader(mode='train', batch_size=TRAIN_CFG['batch_size'], **DATA_CFG)
    val_loader = build_dataloader(mode='val', batch_size=TRAIN_CFG['batch_size'], **DATA_CFG)
    model = OvarianSurvivalModel(**MODEL_CFG)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG['lr'], weight_decay=TRAIN_CFG['weight_decay'])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=lambda r, t, e, g: cox_gate_loss(r, t, e, g, TRAIN_CFG['lambda_reg']),
        epochs=TRAIN_CFG['epochs'],
        patience=TRAIN_CFG['early_stop_patience']
    )
    trainer.train(train_loader, val_loader, device=TRAIN_CFG['device'])
    print("Training completed. Best model saved as best_model.pth")
    