import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class OvarianDataset(Dataset):
    def __init__(self, clinical_path, wsi_dir, us_dir, mode='train'):
        self.clinical_df = pd.read_csv(clinical_path)
        self.wsi_dir = wsi_dir
        self.us_dir = us_dir
        self.mode = mode
        self._encode_clinical_features()
        self.us_transform = self._get_us_transform()

    def _encode_clinical_features(self):
        numeric_feats = self.clinical_df[['age', 'ki67', 'roma']].values
        self.scaler = StandardScaler()
        numeric_scaled = self.scaler.fit_transform(numeric_feats)
        binary_feats = self.clinical_df[['transfer']].values
        figo_encoder = OneHotEncoder(categories=[[1,2,3,4]], sparse_output=False)
        figo_encoded = figo_encoder.fit_transform(self.clinical_df[['figo']].values)
        self.clinical_feats = np.hstack([numeric_scaled, binary_feats, figo_encoded])

    def _get_us_transform(self):
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.RandomHorizontalFlip(p=0.5) if self.mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.clinical_df)

    def __getitem__(self, idx):
        pid = self.clinical_df.iloc[idx]['PatientID']
        wsi_feat = np.load(os.path.join(self.wsi_dir, f'patient_{pid}_wsi.npy')).astype(np.float32)
        us_img = Image.open(os.path.join(self.us_dir, f'patient_{pid}_us.jpg'))
        us_feat = self.us_transform(us_img)
        clinical_feat = self.clinical_feats[idx].astype(np.float32)
        return {
            'wsi': torch.tensor(wsi_feat),
            'us': us_feat,
            'clinical': torch.tensor(clinical_feat),
            'time': torch.tensor(self.clinical_df.iloc[idx]['date'], dtype=torch.float32),
            'event': torch.tensor(self.clinical_df.iloc[idx]['status'], dtype=torch.float32)
        }

def build_dataloader(clinical_path, wsi_dir, us_dir, mode='train', batch_size=8):
    return DataLoader(
        OvarianDataset(clinical_path, wsi_dir, us_dir, mode),
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=4,
        pin_memory=True
    )
    