import torch
import numpy as np
from tqdm import tqdm
from lifelines.utils import concordance_index
from torch.optim.lr_scheduler import CosineAnnealingLR

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False

    def __call__(self, val_cindex, model, save_path='best_model.pth'):
        if val_cindex > self.best_score:
            torch.save(model.state_dict(), save_path)
            self.best_score = val_cindex
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    def __init__(self, model, optimizer, criterion, epochs=100, patience=15):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        self.early_stopping = EarlyStopping(patience=patience)

    def _train_epoch(self, loader, device):
        self.model.train()
        total_loss = 0.0
        risks, times, events = [], [], []
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            risk_score, gate_weights = self.model(batch)
            loss = self.criterion(risk_score, batch['time'], batch['event'], gate_weights)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            risks.append(risk_score.detach().cpu().numpy())
            times.append(batch['time'].cpu().numpy())
            events.append(batch['event'].cpu().numpy())
        avg_loss = total_loss / len(loader)
        cindex = concordance_index(np.concatenate(times), -np.concatenate(risks), np.concatenate(events))
        return avg_loss, cindex

    def _val_epoch(self, loader, device):
        self.model.eval()
        total_loss = 0.0
        risks, times, events = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                risk_score, gate_weights = self.model(batch)
                loss = self.criterion(risk_score, batch['time'], batch['event'], gate_weights)
                total_loss += loss.item()
                risks.append(risk_score.cpu().numpy())
                times.append(batch['time'].cpu().numpy())
                events.append(batch['event'].cpu().numpy())
        avg_loss = total_loss / len(loader)
        cindex = concordance_index(np.concatenate(times), -np.concatenate(risks), np.concatenate(events))
        return avg_loss, cindex

    def train(self, train_loader, val_loader, device='cuda'):
        self.model.to(device)
        for epoch in range(self.epochs):
            train_loss, train_cindex = self._train_epoch(train_loader, device)
            val_loss, val_cindex = self._val_epoch(val_loader, device)
            self.scheduler.step()
            self.early_stopping(val_cindex, self.model)
            print(f'Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | C-index: {train_cindex:.4f} | Val Loss: {val_loss:.4f} | C-index: {val_cindex:.4f}')
            if self.early_stopping.early_stop:
                break
    