import torch
import torch.nn.functional as F
import numpy as np


def cal_anomaly_map(inputs_list, outputs_list, img_size, amap_mode='a'):
    """Compute multi-scale anomaly maps via cosine similarity.
    
    Args:
        inputs_list: list of encoder feature tensors at different scales
        outputs_list: list of decoder feature tensors at same scales
        img_size: target size for upsampling
        amap_mode: 'a' for additive fusion, 'm' for max
    
    Returns:
        anomaly_map: fused anomaly map (H, W) as numpy array
        a_map_list: list of per-scale anomaly maps
    """
    batch_size = inputs_list[0].shape[0]
    a_map_list = []

    for i in range(len(inputs_list)):
        inputs = inputs_list[i]
        outputs = outputs_list[i]

        # Normalize features
        inputs_norm = F.normalize(inputs, p=2, dim=1)
        outputs_norm = F.normalize(outputs, p=2, dim=1)

        # Cosine similarity per pixel
        cos_sim = (inputs_norm * outputs_norm).sum(dim=1, keepdim=True)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # Anomaly score = 1 - cosine_similarity
        anomaly = 1.0 - cos_sim

        # Upsample to target size
        anomaly = F.interpolate(anomaly, size=(img_size, img_size),
                                mode='bilinear', align_corners=False)
        a_map_list.append(anomaly)

    # Fuse all scales
    if amap_mode == 'm':
        anomaly_map = torch.stack(a_map_list, dim=0).max(dim=0)[0]
    else:
        anomaly_map = torch.stack(a_map_list, dim=0).sum(dim=0)

    # Squeeze batch and channel dims
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    a_map_list = [m.squeeze().cpu().numpy() for m in a_map_list]

    return anomaly_map, a_map_list


def evaluation_multi_proj(encoder, bn, decoder, proj_layer, test_dataloader, img_size, amap_mode='a'):
    """Full evaluation loop (stub - not used by active diagnostic)."""
    raise NotImplementedError("Full evaluation requires training pipeline")
