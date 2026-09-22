"""GigaPose retrieval operations adapted from the validated research adapter.

Upstream networks are imported from an explicitly configured checkout.
GigaPose copyright (c) 2023 Van Nguyen Nguyen, MIT; see LICENSES/GigaPose.txt.
"""

from types import SimpleNamespace

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from src.dataloader.template import TemplateSet
from src.models.gigaPose import GigaPose
from src.models.matching import LocalSimilarity
from src.models.network.ae_net import AENet
from src.models.network.ist_net import ISTNet, Regressor
from src.models.network.resnet import ResNet
from src.utils.crop import CropResizePad


def build_transforms():
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )
    crop = CropResizePad(target_size=224)
    return SimpleNamespace(normalize=normalize, crop_transform=crop, inplane_augmentation=False)


def build_model(ckpt_path):
    DINOV2_DIR = DINOV2_SOURCE
    dinov2 = torch.hub.load(DINOV2_DIR, "dinov2_vitl14", source="local", pretrained=False)
    ae_net = AENet(
        model_name="dinov2_vitl14", dinov2_model=dinov2, max_batch_size=64, descriptor_size=1024
    )
    backbone = ResNet(
        config=OmegaConf.create(
            dict(
                n_heads=0,
                input_dim=3,
                input_size=256,
                initial_dim=128,
                block_dims=[128, 192, 256, 512],
                descriptor_size=256,
            )
        )
    )
    regressor = Regressor(
        descriptor_size=256, hidden_dim=256, use_tanh_act=True, normalize_output=True
    )
    ist_net = ISTNet(
        model_name="resnet",
        descriptor_size=256,
        backbone=backbone,
        regressor=regressor,
        max_batch_size=64,
        pretrained_weights=None,
        checkpoint_key="state_dict",
    )
    testing_metric = LocalSimilarity(k=5, sim_threshold=0.5, patch_threshold=3)
    model = GigaPose(
        model_name="large",
        ae_net=ae_net,
        ist_net=ist_net,
        training_loss=None,
        testing_metric=testing_metric,
        optim_config=None,
        log_interval=10**9,
        log_dir=LOG_DIR,
        max_num_dets_per_forward=None,
        test_setting="detection",
    )
    ck = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(ck, strict=False)
    miss_net = [m for m in missing if m.split(".")[0] in ("ae_net", "ist_net")]
    assert not miss_net, f"missing net params: {miss_net[:5]}"
    model = model.to(DEVICE).eval()
    return model


def build_template_dataset(transforms):
    template_config = OmegaConf.create(
        dict(
            dir=str(GP_ROOT / "templates"),
            level_templates=1,
            pose_distribution="all",
            scale_factor=1.0,
            num_templates=162,
            image_name="OBJECT_ID/VIEW_ID.png",
            pose_name="object_poses/OBJECT_ID.npy",
        )
    )
    return TemplateSet(
        root_dir=str(GP_ROOT),
        dataset_name=DS,
        template_config=template_config,
        transforms=transforms,
    )


@torch.no_grad()
def retrieve(model, transforms, tar_img, tar_mask):
    """Replicate GigaPose.eval_retrieval core for a single-object batch.
    tar_img [B,3,224,224] normalized, tar_mask [B,1,224,224]. Returns
    pred_poses [B,k,4,4] (mm-like), scores [B,k], view_ids [B,k]."""
    tdata = model.template_datas[DS]
    precov = model.pose_recovery[DS]
    B = tar_img.shape[0]
    label = torch.full((B,), OBJECT_LABEL, dtype=torch.long, device=DEVICE)
    tar_ae = model.ae_net(tar_img)
    src_ae = tdata.ae_features[label - 1]
    src_masks = tdata.mask[label - 1]
    preds = model.testing_metric.test(
        src_feats=src_ae,
        tar_feat=tar_ae,
        src_masks=src_masks,
        tar_mask=tar_mask,
        max_batch_size=None,
    )
    import pandas as pd

    preds.infos = pd.DataFrame({"label": label.cpu().numpy()})
    k = model.testing_metric.k
    num_patches = preds.src_pts.shape[2]
    pred_scales = torch.zeros(B, k, num_patches, device=DEVICE)
    pred_cosSin = torch.zeros(B, k, num_patches, 2, device=DEVICE)
    idx_sample = torch.arange(B, device=DEVICE)
    tar_ist = model.ist_net.forward_by_chunk(tar_img)
    src_ist_all = tdata.ist_features[label - 1]
    for ik in range(k):
        idx_views = [idx_sample, preds.id_src[:, ik]]
        s, cs = model.ist_net.inference(
            src_feat=src_ist_all[idx_views],
            tar_feat=tar_ist,
            src_pts=preds.src_pts[:, ik],
            tar_pts=preds.tar_pts[:, ik],
        )
        pred_scales[:, ik] = s
        pred_cosSin[:, ik] = cs
    preds.register_tensor("relScale", pred_scales)
    preds.register_tensor("relInplane", pred_cosSin)
    preds = precov.forward_ransac(predictions=preds)
    score = torch.sum(preds.ransac_scores, dim=2) / num_patches
    pred_poses = precov.forward_recovery(
        tar_label=label,
        tar_K=_tar_K(B),
        tar_M=_tar_M,
        pred_src_views=preds.id_src,
        pred_M=preds.M.clone(),
    )
    return (pred_poses, score, preds.id_src)


def _tar_K(B):
    return torch.as_tensor(K_NP, device=DEVICE).unsqueeze(0).repeat(B, 1, 1)


def make_batch(transforms, img_bgr, mask, bbox_xyxy):
    """Build masked square crop batch: rgb*mask -> CropResizePad(224) -> normalize."""
    rgb = torch.from_numpy(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 255.0
    m = torch.from_numpy((mask > 0).astype(np.float32))
    m_rgb = rgb * m[None]
    m_rgba = torch.cat([m_rgb, m[None]], dim=0).unsqueeze(0).to(DEVICE)
    box = torch.as_tensor(bbox_xyxy, dtype=torch.long, device=DEVICE).unsqueeze(0)
    cropped = transforms.crop_transform(box, images=m_rgba)
    rgb_c = cropped["images"][:, :3]
    mask_c = cropped["images"][:, -1]
    tar_img = transforms.normalize(rgb_c)
    return (tar_img, mask_c, cropped["M"])


DEVICE = "cuda"
DS = None
GP_ROOT = None
DINOV2_SOURCE = None
LOG_DIR = None
OBJECT_LABEL = 1
K_NP = None
_tar_M = None
