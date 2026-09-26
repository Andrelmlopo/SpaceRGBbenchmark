from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .backends import pose_transform


class RGBTrackBackend:
    def __init__(self, settings, sequence, output):
        self.settings, self.sequence = settings, sequence
        source = Path(settings["rgbtrack_source"])
        os.chdir(source)
        sys.path.insert(0, str(source))
        import estimater as native
        import nvdiffrast.torch as dr
        import torch
        import trimesh
        import xmem_wrapper as xmem
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        from learning.training.predict_score import ScorePredictor
        from tools import binary_search_depth
        from Utils import set_seed

        self.torch, self.xmem, self.initialize = torch, xmem, binary_search_depth
        set_seed(settings.get("seed", 0))
        self.mesh = trimesh.load(sequence.file(sequence.config["mesh"]), force="mesh")
        self.mesh.apply_scale({"m": 1, "cm": 0.01, "mm": 0.001}[sequence.config["mesh_units"]])
        if isinstance(self.mesh.visual, trimesh.visual.texture.TextureVisuals) and not hasattr(
            self.mesh.visual.material, "image"
        ):
            material = self.mesh.visual.material
            image = getattr(material, "baseColorTexture", None)
            if image is None:
                raise ValueError("Textured PBR mesh lacks baseColorTexture")
            converted = trimesh.visual.material.SimpleMaterial(
                image=image, diffuse=getattr(material, "baseColorFactor", None)
            )
            self.mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=self.mesh.visual.uv.copy(), material=converted
            )
        self.est = FoundationPose(
            self.mesh.vertices,
            self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug=0,
            debug_dir=str(Path(output) / "debug"),
            glctx=dr.RasterizeCudaContext(),
        )
        self.native_render = native.render_cad_mask
        native.render_cad_mask = lambda p, m, K, w=640, h=480: self.native_render(p, m, K, w=h, h=w)
        self.net = xmem.XMem(xmem.config, str(source / "XMem/saves/XMem.pth")).eval().cuda()
        self.processor = xmem.InferenceCore(self.net, config=xmem.config)
        self.processor.set_all_labels([1])
        self.initialized = False
        self.transform = pose_transform(sequence, "source_mesh")

    def step(self, index, command):
        torch, xm, sequence, est = self.torch, self.xmem, self.sequence, self.est
        rgb, K = sequence.rgb(index), sequence.intrinsics(index)
        tensor, _ = xm.image_to_torch(rgb, device="cuda")
        phase = "tracking"
        if not self.initialized:
            mask = sequence.localization(index)
            if mask is None:
                return dict(pose=None, phase="missing", failure="no_initial_detection")
            labels = xm.index_numpy_to_one_hot_torch(mask.astype(np.uint8), 2).cuda()
            with torch.cuda.amp.autocast():
                self.processor.step(tensor, labels[1:])
            # Fixed geometry-derived depth bracket, independent of GT pose/range.
            box = sequence.bbox(index)
            diameter = float(np.linalg.norm(np.ptp(self.mesh.vertices, axis=0)))
            proxy = float(max(K[0, 0], K[1, 1]) * diameter / max(box[2] - box[0], box[3] - box[1]))
            bounds = self.settings.get("depth_search_m", [max(proxy * 0.25, 0.01), proxy * 4])
            pose = self.initialize(
                est,
                self.mesh,
                rgb,
                mask,
                K,
                depth_min=bounds[0],
                depth_max=bounds[1],
                w=sequence.width,
                h=sequence.height,
                debug=False,
                ycb=sequence.config["dataset"] == "ycbv",
                iteration=5,
            )
            centered = est.pose_last.detach().cpu().numpy().reshape(4, 4)
            if centered[2, 3] <= 0 or not np.isfinite(centered).all():
                raise ValueError("RGB-only initializer returned invalid depth")
            est.last_depth = np.full(mask.shape, centered[2, 3], np.float32)
            est.xyz = est.pose_last.reshape(4, 4)[:3, 3]
            self.initialized, phase = True, "initialization"
        else:
            with torch.cuda.amp.autocast():
                probabilities = self.processor.step(tensor)
            mask = xm.torch_prob_to_numpy_mask(probabilities)
            recovering = est.track_good is False and int(mask.sum()) > 40
            pose = est.track_one_new_without_depth(
                rgb=rgb, K=K, mask=mask, iteration=self.settings.get("track_refine_iterations", 1)
            )
            if recovering:
                original = est.pose_last.detach().cpu().numpy().reshape(4, 4)
                to_center = est.get_tf_to_centered_mesh().detach().cpu().numpy().reshape(4, 4)
                centered = original @ np.linalg.inv(to_center)
                est.pose_last = torch.as_tensor(centered, dtype=torch.float32, device="cuda")
                est.xyz = est.pose_last[:3, 3]
                est.last_depth = np.full(mask.shape, centered[2, 3], np.float32)
                pose, phase = original, "recovery"
        return dict(pose=(np.asarray(pose).reshape(4, 4) @ self.transform).tolist(), phase=phase)
