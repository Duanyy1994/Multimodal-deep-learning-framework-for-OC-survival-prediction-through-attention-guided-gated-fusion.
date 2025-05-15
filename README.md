# Multimodal-deep-learning-framework-for-OC-survival-prediction-through-attention-guided-gated-fusion.
Ovarian Cancer: Deep Learning for Survival Prediction and Risk Stratification Using Multi-Modal Data
# Ovarian Cancer Multimodal Survival Prediction Model

## 1. Project Overview
This project builds a survival prediction model by fusing ultrasound images, pathology WSI features (extracted via CHIEF), and clinical data. Core components include:
- **Ultrasound Feature Extraction**: Uses a pretrained ResNet-50 with frozen convolutional layers.
- **Clinical Feature Encoding**: Maps 8-dimensional raw clinical features to 256-dimensional embeddings.
- **Cross-Modal Attention**: Adapts to arbitrary WSI feature dimensions (e.g., 768 from CHIEF).
- **Gated Fusion**: Dynamically selects critical features via a gating mechanism.


## 2. CHIEF Model Declaration
WSI features are extracted using the CHIEF model. The original paper title is:  
"A pathology foundation model for cancer diagnosis and prognosis prediction"  
Code link: https://github.com/hms-dbmi/CHIEF (replace with the real link).


## 3. Environment Requirementstorch>=1.9.0
torchvision>=0.10.0
pandas>=1.3.0
numpy>=1.21.0
lifelines>=0.26.0
tqdm>=4.62.0
Pillow>=8.3.0
scikit-learn>=1.0.0

## 4. Training Steps
1. Install dependencies:  
   `pip install -r requirements.txt`

2. Organize data:  
   Place clinical data (`clinical.csv`), ultrasound images (`ultrasound_images/`), and CHIEF-extracted WSI features (`wsi_features/`) in the `data` directory.

3. Start training:  
   `python run.py`

The best model will be saved as `best_model.pth` upon completion.
    
