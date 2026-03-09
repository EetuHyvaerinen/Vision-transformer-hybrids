import torch
import torch.nn as nn
from config import EMBED_DIM, N_LAYERS, N_ATTENTION_HEADS, FORWARD_MUL, N_CLASSES, DROPOUT, N_CHANNELS, IMAGE_SIZE, PATCH_SIZE
from models.cnn import CNN_feature_extractor
from models.vit import VisionTransformer, ViT_feature_extractor

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = CNN_feature_extractor()
        self.vit_model = VisionTransformer(n_channels=64, embed_dim=EMBED_DIM, 
                          n_layers=N_LAYERS, n_attention_heads=N_ATTENTION_HEADS, 
                          forward_mul=FORWARD_MUL, image_size=7, 
                          patch_size=7, n_classes=N_CLASSES, dropout=DROPOUT)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.vit_model(features)
        return output

class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extractor = CNN_feature_extractor()
        self.vit_feature_extractor = ViT_feature_extractor(n_channels=N_CHANNELS, embed_dim=EMBED_DIM, 
                          n_layers=N_LAYERS, n_attention_heads=N_ATTENTION_HEADS, 
                          forward_mul=FORWARD_MUL, image_size=IMAGE_SIZE, 
                          patch_size=PATCH_SIZE, n_classes=N_CLASSES, dropout=DROPOUT)

        self.fc_fusion = nn.Sequential(
            nn.Linear(3136 + 64, 128),  
            nn.ReLU(),
            nn.Linear(128, N_CLASSES)
        )
        
    def forward(self, x):
        cnn_features = self.cnn_feature_extractor(x)
        cnn_features = torch.flatten(cnn_features, start_dim=1)
        
        vit_features = self.vit_feature_extractor(x)
        vit_features = vit_features[:, 0, :]
        
        fused_features = torch.cat((cnn_features, vit_features), dim=1)
        output = self.fc_fusion(fused_features)
        return output