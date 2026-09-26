from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..io import read_json, sha256
from ..schema import valid_pose


def pose_transform(sequence, mode):
    if mode == "source_mesh":
        return np.linalg.inv(np.asarray(sequence.config["raw_to_object"]))
    if mode == "benchmark":
        return np.eye(4)
    raise ValueError("Declare output pose frame as source_mesh or benchmark")


class PicoBackend:
    def __init__(self, settings, sequence, output):
        from pico_hat.pico import PicoPose

        self.sequence = sequence
        metadata = read_json(Path(settings["templates"]) / "metadata.json")
        if metadata["mesh_sha256"] != sha256(sequence.file(sequence.config["mesh"])):
            raise ValueError("Pico templates belong to a different CAD asset")
        if metadata["mesh_units"] != sequence.config["mesh_units"]:
            raise ValueError("Pico template mesh units differ")
        self.transform = pose_transform(sequence, settings.get("pico_pose_frame", "source_mesh"))
        self.model = PicoPose(
            settings["pico_source"], settings["pico_checkpoint"], settings["templates"]
        )

    def step(self, index, command):
        sequence = self.sequence
        poses, scores = self.model.predict(
            sequence.rgb(index), sequence.localization(index), sequence.intrinsics(index)
        )
        return dict(poses=(poses @ self.transform).tolist(), scores=scores.tolist())


class DroidBackend:
    def __init__(self, settings, sequence, output):
        from pico_hat.droid import DroidMotion

        self.sequence = sequence

        # Preserve separate silhouette inputs for the relative-motion branch.
        class MotionSequence:
            def __getattr__(self, key):
                return getattr(sequence, key)

            def localization(self, index):
                from PIL import Image

                path = sequence.frames[index].get("slam_mask")
                if path:
                    with Image.open(sequence.file(path)) as image:
                        mask = np.asarray(image) != 0
                    if mask.shape != (sequence.height, sequence.width):
                        raise ValueError("SLAM mask dimensions differ from RGB")
                    return mask if mask.any() else None
                return sequence.localization(index)

        self.model = DroidMotion(
            settings["droid_source"],
            settings["droid_checkpoint"],
            MotionSequence(),
            settings.get("droid_buffer", 1024),
        )

    def step(self, index, command):
        # DROID accepts per-frame calibration, including varying native intrinsics.
        K = self.sequence.intrinsics(index)
        values = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        intrinsics = self.model.torch.as_tensor(values)
        intrinsics[0::2] *= self.model.width / self.sequence.width
        intrinsics[1::2] *= self.model.height / self.sequence.height
        self.model.intrinsics = intrinsics
        return dict(pose=self.model.step(index).tolist())


class MegaBackend:
    def __init__(self, settings, sequence, output):
        self.settings, self.sequence = settings, sequence
        sys.path.insert(0, str(Path(settings["mega_source"]) / "src"))
        os.environ["MEGAPOSE_DATA_DIR"] = settings["mega_data"]
        import torch
        from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
        from megapose.utils.load_model import load_named_model

        torch.multiprocessing.set_start_method("spawn", force=True)
        self.transform = pose_transform(sequence, settings["mega_pose_frame"])
        self.label = "target"
        objects = RigidObjectDataset(
            [
                RigidObject(
                    label=self.label,
                    mesh_path=sequence.file(sequence.config["mesh"]),
                    mesh_units=sequence.config["mesh_units"],
                )
            ]
        )
        self.model = (
            load_named_model(
                settings.get("mega_model", "megapose-1.0-RGB-multi-hypothesis"),
                objects,
                n_workers=settings.get("render_workers", 1),
                bsz_images=32,
            )
            .cuda()
            .eval()
        )

    def step(self, index, command):
        import pandas as pd
        import torch
        from megapose.inference.types import ObservationTensor
        from megapose.utils.tensor_collection import PandasTensorCollection

        s = self.sequence
        rgb, K, box = s.rgb(index), s.intrinsics(index), s.bbox(index)
        padding = self.settings.get("bbox_padding", 0.0)
        if padding:
            box = np.asarray(box, dtype=np.float32)
            center = (box[:2] + box[2:]) / 2
            extent = (box[2:] - box[:2]) * (1 + padding) / 2
            box = np.r_[
                np.maximum(center - extent, 0), np.minimum(center + extent, [s.width, s.height])
            ]
        observation = ObservationTensor.from_numpy(rgb, None, K).cuda()
        info = dict(label=self.label, batch_im_id=0, instance_id=0, score=1.0)
        detection = PandasTensorCollection(
            infos=pd.DataFrame([info]), bboxes=torch.as_tensor(box, dtype=torch.float32)[None]
        ).cuda()
        kwargs = dict(
            detections=detection,
            n_refiner_iterations=self.settings.get("refiner_iterations", 5),
            n_pose_hypotheses=self.settings.get("candidates", 10),
        )
        if command.get("priors") is not None:
            # Priors arrive in the benchmark object frame, converted back for the renderer.
            priors = np.asarray(command["priors"]) @ np.linalg.inv(self.transform)
            priors = priors[[valid_pose(p) for p in priors]]
            if len(priors) == 0:
                return dict(poses=[], scores=[])
            infos = pd.DataFrame([{**info, "instance_id": i} for i in range(len(priors))])
            kwargs["coarse_estimates"] = PandasTensorCollection(
                infos=infos, poses=torch.as_tensor(priors, dtype=torch.float32)
            ).cuda()
            kwargs["n_pose_hypotheses"] = 1
        with torch.no_grad():
            _, extra = self.model.run_inference_pipeline(observation, **kwargs)
        scored = extra["scoring"]["preds"]
        poses, logits = scored.poses.cpu().numpy(), scored.infos["pose_logit"].values
        order = np.argsort(-logits)[: self.settings.get("candidates", 10)]
        return dict(poses=(poses[order] @ self.transform).tolist(), scores=logits[order].tolist())

    def close(self):
        renderers = {
            id(m.renderer): m.renderer for m in (self.model.coarse_model, self.model.refiner_model)
        }
        for renderer in renderers.values():
            renderer.stop()


class GigaBackend:
    def __init__(self, settings, sequence, output):
        source = Path(settings["giga_source"])
        sys.path[:0] = [str(source), str(source / "src")]
        from . import giga_ops as gp

        self.gp, self.sequence = gp, sequence
        gp.DINOV2_SOURCE = settings["dinov2_source"]
        gp.LOG_DIR = str(Path(output) / "giga_logs")
        gp.DS = settings["giga_dataset"]
        gp.GP_ROOT = Path(settings["giga_root"])
        gp.OBJECT_LABEL = settings.get("giga_object_label", sequence.object_id)
        self.transform = pose_transform(sequence, settings["giga_pose_frame"])
        self.transforms = gp.build_transforms()
        self.model = gp.build_model(settings["giga_checkpoint"])
        self.model.template_datasets = {gp.DS: gp.build_template_dataset(self.transforms)}
        self.model.test_dataset_name = gp.DS
        import torch

        with torch.no_grad():
            self.model.set_template_data(gp.DS)

    def step(self, index, command):
        gp, s = self.gp, self.sequence
        gp.K_NP = s.intrinsics(index).astype(np.float32)
        image, mask, gp._tar_M = gp.make_batch(
            self.transforms, s.rgb(index)[..., ::-1].copy(), s.localization(index), s.bbox(index)
        )
        poses, scores, _ = gp.retrieve(self.model, self.transforms, image, mask)
        poses = poses[0].detach().cpu().numpy().astype(float)
        poses[:, :3, 3] /= 1000
        return dict(
            poses=(poses @ self.transform).tolist(), scores=scores[0].cpu().numpy().tolist()
        )


def make_backend(name, settings, sequence, output):
    if name == "rgbtrack":
        from .rgbtrack import RGBTrackBackend

        return RGBTrackBackend(settings, sequence, output)
    return {"pico": PicoBackend, "droid": DroidBackend, "mega": MegaBackend, "giga": GigaBackend}[
        name
    ](settings, sequence, output)
