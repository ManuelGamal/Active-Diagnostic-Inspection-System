# Model (Engineer 2)

## Architecture

Supervised binary classifier:

1. `ResNet50Encoder` (timm `resnet50`, ImageNet-pretrained)
2. `ClassificationHead` (`AdaptiveAvgPool2d(1)` -> `Dropout(0.3)` -> `Linear(2048, 2)`)

Input: `(B, 3, 224, 224)`  
Output logits: `(B, 2)` (normal vs defect)

## Why this design

- **ResNet-50 + ImageNet pretraining**: strong transfer baseline for industrial textures and limited-label regimes.
- **Focal Loss**: reduces dominance of easy/majority samples in imbalanced data.
- **AdamW + cosine schedule**: stable optimization and smooth LR decay for short/medium runs.

## Frozen defaults

- Backbone: `resnet50`
- Loss: focal (`gamma=2.0`, `alpha=0.25`)
- Optimizer: `AdamW(lr=1e-4, weight_decay=1e-4)`
- Scheduler: `CosineAnnealingLR(T_max=epochs)`
- Precision recommendation on T4: `16-mixed` (fp16)

## Files

- `src/models/encoder.py`
- `src/models/classification_head.py`
- `src/models/losses.py`
- `src/models/lit_module.py`
- `src/models/gradcam.py`
- `src/models/export_onnx.py`

## ONNX export

```bash
bash scripts/export_onnx.sh <checkpoint.ckpt> <output.onnx>
```

The exporter runs a parity check against PyTorch on 50 random inputs and fails if `max_abs_diff >= 1e-4`.

## Grad-CAM usage

```python
from src.models.gradcam import get_gradcam, overlay_cam

heatmap = get_gradcam(model, image_tensor)      # image_tensor: (1,3,224,224)
overlay = overlay_cam(image_np, heatmap)        # image_np: (224,224,3), in [0,1]
```

## Extending to other backbones

To switch backbones, replace `timm.create_model("resnet50", ...)` in `ResNet50Encoder` with another timm model and update `feature_dim` to match the new channel count.

