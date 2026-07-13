# ============================ agrrm_model.py ============================
# AGRRM 模型文件（
# ========================================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ============================================
# 常量定义
# ============================================
INPUT_SIZE = 448
PART_NAMES = [
    "forehead_left", "forehead_center", "forehead_right",
    "eye_left", "eye_right",
    "cheek_left", "cheek_center", "cheek_right",
    "nose", "mouth_left", "mouth_right",
    "ear_left", "ear_right"
]


# ============================================
# PartAttention 模块
# ============================================
class PartAttention(nn.Module):
    def __init__(self, in_channels, mid_channels=256, feat_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.projector = nn.Sequential(
            nn.Linear(in_channels, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_map, region_mask=None):
        attn_map = self.attention(feat_map)
        attn_loss = None
        if region_mask is not None:
            if region_mask.shape[2:] != attn_map.shape[2:]:
                region_mask = F.interpolate(region_mask, size=attn_map.shape[2:], mode='nearest')
            attn_loss = F.binary_cross_entropy(attn_map, region_mask)

        weighted_feat = feat_map * attn_map
        p = 3
        part_feat = (weighted_feat.clamp(min=1e-6).pow(p).mean(dim=[2, 3])).pow(1 / p)
        part_feat = self.projector(part_feat)
        return part_feat, attn_map, attn_loss


# ============================================
# RRM(Region Response Module)
# ============================================
class RRM(nn.Module):
    def __init__(self, in_channels, num_parts=13):
        super().__init__()
        self.num_parts = num_parts

        self.angle_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64)
        )

        self.feat_processor = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.visibility_predictor = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_parts),
            nn.Sigmoid()
        )

    def forward(self, feat_map, angle):
        angle_feat = self.angle_encoder(angle)
        img_feat = self.feat_processor(feat_map).flatten(1)
        combined = torch.cat([img_feat, angle_feat], dim=1)
        return self.visibility_predictor(combined)


# ============================================
# EfficientNet-B3 + AGRRM ReID model
# ============================================
class EfficientNetB3_AGRRM_ReID(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()

        full_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = full_model.features

        dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = self.features(dummy)
        self.backbone_out_channels = out.shape[1]

        # Global Feature Module
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(self.backbone_out_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Region-Supervised Part Attention Learning
        self.part_attentions = nn.ModuleDict()
        for part_name in PART_NAMES:
            self.part_attentions[part_name] = PartAttention(
                in_channels=self.backbone_out_channels,
                mid_channels=min(256, self.backbone_out_channels // 2),
                feat_dim=256
            )

        # RRM
        self.rrm = RRM(in_channels=self.backbone_out_channels, num_parts=13)

        # Pose-aware Feature Fusion
        fusion_dim = 512 + 256 * len(PART_NAMES) + 13
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.bnneck = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)

        if weights is not None:
            self.load_weights(weights)

    def load_weights(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        self.load_state_dict(ckpt, strict=False)
        print(f"✅ Weight success: {checkpoint_path}")

    def forward_features(self, x):
        return self.features(x)

    def forward_parts(self, feat_map, region_masks=None):
        part_features = []
        attention_losses = []
        for i, (part_name, attn_module) in enumerate(self.part_attentions.items()):
            if region_masks is not None:
                part_mask = region_masks[:, i:i + 1, :, :]
                if part_mask.shape[2:] != feat_map.shape[2:]:
                    part_mask = F.interpolate(part_mask, size=feat_map.shape[2:], mode='nearest')
            else:
                part_mask = None
            part_feat, _, attn_loss = attn_module(feat_map, part_mask)
            part_features.append(part_feat)
            if attn_loss is not None:
                attention_losses.append(attn_loss)
        attn_loss = torch.stack(attention_losses).mean() if attention_losses else torch.tensor(0.0, device=feat_map.device)
        return part_features, attn_loss

    def forward(self, x, region_masks=None, angle=None):
        feat_map = self.forward_features(x)

        global_feat = self.global_pool(feat_map).flatten(1)
        global_feat = self.global_fc(global_feat)

        part_features, attn_loss = self.forward_parts(feat_map, region_masks)
        part_concat = torch.cat(part_features, dim=1)

        if angle is not None:
            region_responses = self.rrm(feat_map, angle)
        else:
            region_responses = torch.zeros(x.size(0), 13, device=x.device)

        combined = torch.cat([global_feat, part_concat, region_responses], dim=1)
        final_feat = self.fusion(combined)
        bn_feat = self.bnneck(final_feat)
        logits = self.classifier(bn_feat)

        return logits, final_feat, global_feat, part_features, attn_loss, region_responses

    def extract_features(self, x, angle=None):
        """推理用：返回 L2 归一化的特征向量"""
        feat_map = self.forward_features(x)

        global_feat = self.global_pool(feat_map).flatten(1)
        global_feat = self.global_fc(global_feat)

        part_features, _ = self.forward_parts(feat_map, region_masks=None)
        part_concat = torch.cat(part_features, dim=1)

        if angle is not None:
            region_responses = self.rrm(feat_map, angle)
        else:
            region_responses = torch.zeros(x.size(0), 13, device=x.device)

        combined = torch.cat([global_feat, part_concat, region_responses], dim=1)
        final_feat = self.fusion(combined)
        bn_feat = self.bnneck(final_feat)

        return F.normalize(bn_feat, dim=1)

