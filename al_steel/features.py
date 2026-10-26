"""Frozen pretrained backbones from timm, used as fixed feature extractors."""
import numpy as np
import timm
import torch

# Short name -> timm model id. DINOv2 is self-supervised, ResNet-18 is supervised on ImageNet.
BACKBONES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "resnet18": "resnet18.a1_in1k",
}


def build_backbone(name, img_size=224):
    kwargs = {"img_size": img_size} if name.startswith("dinov2") else {}
    model = timm.create_model(BACKBONES[name], pretrained=True, num_classes=0, **kwargs)
    model.eval()
    cfg = timm.data.resolve_data_config({"input_size": (3, img_size, img_size)}, model=model)
    cfg["crop_pct"] = 1.0  # keep the whole 200x200 image, defects can sit at the border
    return model, timm.data.create_transform(**cfg)


@torch.no_grad()
def extract_features(images, name, batch_size=32):
    """Return an (N, D) float32 array of pooled backbone features."""
    model, transform = build_backbone(name)
    feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([transform(im) for im in images[i:i + batch_size]])
        feats.append(model(batch).numpy())
    return np.concatenate(feats).astype(np.float32)
