"""
Frozen ResNet18 backbone for visual feature extraction.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FeatureExtractor(nn.Module):
    """
    Frozen ResNet18 backbone that extracts multi-scale features.
    Extracts features from layer2 (stride 8) and layer3 (stride 16).
    """
    
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
        else:
            resnet = resnet18(weights=None)
        
        # Extract layers (remove avgpool and fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # Output: 64 channels, stride 4
        self.layer2 = resnet.layer2  # Output: 128 channels, stride 8
        self.layer3 = resnet.layer3  # Output: 256 channels, stride 16
        # We don't use layer4 to save computation
        
        # Freeze all parameters
        self._freeze_backbone()
        
        # Set to eval mode permanently
        self.eval()
        
    def _freeze_backbone(self):
        """
        Freeze all parameters in the backbone.
        """
        for param in self.parameters():
            param.requires_grad = False
            
        print("[FeatureExtractor] All parameters frozen. Total params:", 
              sum(p.numel() for p in self.parameters()))
    
    def train(self, mode=True):
        """
        Override train method to keep backbone in eval mode.
        This ensures BatchNorm uses running statistics, not batch statistics.
        """
        # Keep this module in eval mode
        super(FeatureExtractor, self).train(False)
        return self
    
    def forward(self, x):
        """
        Forward pass to extract multi-scale features.
        
        Args:
            x: (B, 3, H, W) - Input RGB images
            
        Returns:
            features: list of [layer2_feat, layer3_feat]
                layer2_feat: (B, 128, H/8, W/8)
                layer3_feat: (B, 256, H/16, W/16)
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Blocks
        x = self.layer1(x)  # stride 4
        feat_layer2 = self.layer2(x)  # stride 8, 128 channels
        feat_layer3 = self.layer3(feat_layer2)  # stride 16, 256 channels
        
        return [feat_layer2, feat_layer3]


if __name__ == "__main__":
    # Test the feature extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FeatureExtractor(pretrained=True).to(device)
    
    # Test input
    x = torch.randn(2, 3, 512, 512).to(device)
    
    with torch.no_grad():
        features = model(x)
    
    print("Input shape:", x.shape)
    for i, feat in enumerate(features):
        print(f"Feature {i} shape:", feat.shape)
    
    # Check if gradients are disabled
    print("\nGradient check:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")
