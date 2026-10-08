#!/usr/bin/env python3
"""SimLingo model runner, free of ROS and of CARLA.

Everything model-specific from ``simlingo_ros/simlingo_node.py`` lives here:
checkpoint + hydra config loading, the InternVL2 image pipeline, the prompt
template, ``DrivingInput`` assembly and the forward pass.  The ROS node only
gathers sensor data and converts frames.

Threading contract: ``load()`` and every ``infer()`` must run in the *same* OS
thread.  On Jetson iGPU the cuBLAS/cuDNN handles are created lazily in the
thread that first uses them and are not usable from another thread; the node
therefore drives this class from a single-worker ThreadPoolExecutor, exactly
as the reference node does.

Units: the model was trained in CARLA at full scale.  This class takes speed in
m/s and target points in metres *as the model expects them*; scaling a 1:10 car
into that domain is the caller's job (see ``world_scale`` in the node).
"""

from __future__ import annotations

import importlib.util
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# Camera geometry the model was trained with (config_simlingo.GlobalConfig).
# The intrinsics are computed for the 448x448 tile, the way agent_simlingo.py
# does it (get_camera_intrinsics(W, H, 110) with W, H the tile size), and the
# extrinsics are the CARLA mounting position with no rotation.  Both are fixed
# properties of the checkpoint, not of the physical camera; a real camera is
# made to *look like* this one by the frame formatting below.
TRAIN_CAM_FOV_DEG = 110.0
TRAIN_CAM_EXTRINSIC_XYZ = (-1.5, 0.0, 2.0)
TRAIN_IMAGE_W = 1024
TRAIN_IMAGE_H = 512
# Fraction of the image height removed from the bottom (the car's hood in
# CARLA): agent_simlingo.py:375 crops ``h - h*4.8//16``.
TRAIN_BOTTOM_CROP_FRAC = 4.8 / 16.0
# Seconds between predicted waypoints: carla_fps=20, wp_dilation=1,
# data_save_freq=5  ->  20/(1*5) = 4 Hz.
WAYPOINT_DT = 0.25

_SPECIAL_TOKENS = [
    "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", "<TARGET_POINT>",
]


@dataclass
class InferenceResult:
    route: np.ndarray          # [20, 2] model ego frame (x forward, y RIGHT), model metres
    speed_wps: np.ndarray      # [10, 2] model ego frame, model metres
    language: str
    model_ms: float

    @property
    def desired_speed(self) -> float:
        """Scalar target speed exactly as agent_simlingo.py::control_pid derives it.

        one_second = carla_fps // (wp_dilation * data_save_freq) = 4, half = 2:
        distance travelled over 0.5 s of speed waypoints, doubled -> m/s.
        """
        return float(np.linalg.norm(self.speed_wps[0] - self.speed_wps[2]) * 2.0)


def format_camera_frame(rgb: np.ndarray,
                        out_w: int = TRAIN_IMAGE_W,
                        out_h: int = TRAIN_IMAGE_H,
                        bottom_crop_frac: float = TRAIN_BOTTOM_CROP_FRAC) -> np.ndarray:
    """Make a real camera frame look like a CARLA training frame.

    1. centre-crop to the training aspect ratio (2:1) -- a 16:9 webcam frame
       loses a strip top and bottom rather than being squashed;
    2. resize to the training resolution;
    3. remove the bottom ``bottom_crop_frac`` of the height, as training did for
       the hood.  The F1TENTH has no hood, but the model has only ever seen
       1024x358-shaped input at this FOV, and the InternVL2 tiler picks its tile
       layout from the aspect ratio, so the crop is kept by default.
    """
    h, w = rgb.shape[:2]
    target_aspect = out_w / out_h
    if w / h > target_aspect:                       # too wide: crop the sides
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        rgb = rgb[:, x0:x0 + new_w]
    elif w / h < target_aspect:                     # too tall: crop top/bottom
        new_h = int(round(w / target_aspect))
        y0 = (h - new_h) // 2
        rgb = rgb[y0:y0 + new_h, :]
    if rgb.shape[1] != out_w or rgb.shape[0] != out_h:
        interp = cv2.INTER_AREA if rgb.shape[1] > out_w else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=interp)
    # Same arithmetic as training (dataset_base.py:467) and the agent
    # (agent_simlingo.py: ``h - (h * 4.8) // 16``): the floor applies to the
    # cropped-off rows, not to the kept rows, so 512 -> 359 rows, not 358.
    crop_h = int(out_h - (out_h * bottom_crop_frac * 16.0) // 16.0)
    return np.ascontiguousarray(rgb[:crop_h])


def jpeg_roundtrip(rgb: np.ndarray, quality: int = 95) -> np.ndarray:
    """Training frames were stored as JPEG; reproduce the codec artefacts."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class SimLingoRunner:
    """Loads the checkpoint once and turns (image, speed, target points) into waypoints."""

    def __init__(self, checkpoint_path: str, simlingo_path: str,
                 fast_inference: bool = True,
                 use_cot: bool = False,
                 camera_fov_deg: float = TRAIN_CAM_FOV_DEG,
                 logger: Optional[logging.Logger] = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.simlingo_path = simlingo_path
        self.fast_inference = fast_inference
        # Chain-of-thought ("thinking") prompt, as agent_simlingo.py with config.use_cot:
        # the model first writes a commentary on what to do and why, then the waypoints
        # are decoded after that text.  Each generated token costs ~10 ms with
        # fast_inference (~135 ms without), so a sentence adds a few hundred ms per plan.
        self.use_cot = use_cot
        self.camera_fov_deg = camera_fov_deg
        self.log = logger or logging.getLogger("simlingo_runner")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._cfg = None
        self._transform = None
        self._conv_module = None
        self._num_image_token = 256
        self._tp_token_id = None
        self._K = None
        self._E = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ── loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        import hydra
        from omegaconf import OmegaConf
        from transformers import AutoConfig, AutoProcessor

        ckpt = Path(self.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        # <session>/<version>/<epoch>/weights.ckpt -> <session>/.hydra/config.yaml
        cfg_path = ckpt.parent.parent.parent / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Expected hydra config at {cfg_path}")
        self.log.info(f"Loading SimLingo checkpoint {ckpt}")
        cfg = OmegaConf.load(cfg_path)
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        self._cfg = cfg

        vlm_variant = cfg.model.vision_model.variant            # OpenGVLab/InternVL2-1B
        vlm_cache = str(Path(self.simlingo_path) / "pretrained" / vlm_variant.split("/")[1])

        processor = AutoProcessor.from_pretrained(vlm_variant, trust_remote_code=True, cache_dir=vlm_cache)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        tokenizer.add_special_tokens({"additional_special_tokens": _SPECIAL_TOKENS})
        tokenizer.padding_side = "left"
        self._tokenizer = tokenizer
        self._tp_token_id = tokenizer.convert_tokens_to_ids("<TARGET_POINT>")

        vlm_cfg = AutoConfig.from_pretrained(vlm_variant, trust_remote_code=True, cache_dir=vlm_cache)
        image_size = vlm_cfg.force_image_size or vlm_cfg.vision_config.image_size
        patch_size = vlm_cfg.vision_config.patch_size
        self._num_image_token = int((image_size // patch_size) ** 2 * (vlm_cfg.downsample_ratio ** 2))

        conv_py = Path(vlm_cache) / "conversation.py"
        if not conv_py.exists():
            from huggingface_hub import hf_hub_download
            conv_py = Path(hf_hub_download(repo_id=vlm_variant, filename="conversation.py"))
        spec = importlib.util.spec_from_file_location("_simlingo_conv", str(conv_py))
        conv_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conv_module)
        self._conv_module = conv_module

        from simlingo_training.utils.internvl2_utils import build_transform
        self._transform = build_transform(input_size=448)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self._model = hydra.utils.instantiate(
            cfg.model, cfg_data_module=cfg.data_module, processor=processor,
            cache_dir=vlm_cache, _recursive_=False,
        ).to(self.device)
        torch.set_default_dtype(default_dtype)
        self._model.load_state_dict(torch.load(str(ckpt), map_location=self.device))
        self._model.eval()

        if self.fast_inference:
            try:
                from simlingo_training.models.fast_inference import optimize_for_inference
                self.log.info(f"fast_inference: {optimize_for_inference(self._model)}")
            except Exception as exc:                      # the stock model still drives
                self.log.error(f"fast_inference failed ({type(exc).__name__}: {exc}); using stock model")

        # Warm-up in this thread so the CUDA handles belong to it.
        torch.cuda.empty_cache()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _d = torch.zeros(1, 1, device=self.device, dtype=torch.bfloat16)
            torch.mm(_d, _d)
        torch.cuda.synchronize()

        # Fixed calibration (see module docstring).
        W = H = 448
        focal = W / (2.0 * math.tan(self.camera_fov_deg * math.pi / 360.0))
        self._K = torch.tensor([[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
                               dtype=torch.float32).unsqueeze(0).to(self.device)
        E = torch.eye(4, dtype=torch.float32)
        E[:3, 3] = torch.tensor(TRAIN_CAM_EXTRINSIC_XYZ)
        self._E = E.unsqueeze(0).to(self.device)
        self.log.info("SimLingo model ready.")

    # ── image preprocessing (any thread) ─────────────────────────────────────

    def preprocess(self, rgb_cropped: np.ndarray):
        """InternVL2 dynamic tiling + normalisation.  Returns ([1,1,P,3,448,448], P)."""
        from simlingo_training.utils.internvl2_utils import dynamic_preprocess
        tiles = dynamic_preprocess(
            PILImage.fromarray(rgb_cropped), image_size=448,
            use_thumbnail=self._cfg.model.vision_model.use_global_img, max_num=2,
        )
        pixel_values = torch.stack([self._transform(t) for t in tiles])
        return pixel_values.unsqueeze(0).unsqueeze(0), pixel_values.shape[0]

    # ── inference (loader thread only) ───────────────────────────────────────

    def infer(self, processed_image: torch.Tensor, num_patches: int,
              speed_mps: float, target_points: np.ndarray) -> InferenceResult:
        """target_points: [2, 2] float32, model ego frame (x forward, y right), model metres."""
        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel

        question = "What should the ego do next?" if self.use_cot else "Predict the waypoints."
        prompt = (f"Current speed: {round(float(speed_mps), 1)} m/s. "
                  f"Target waypoint: <TARGET_POINT><TARGET_POINT>. {question}")
        template = self._conv_module.get_conv_template("internlm2-chat")
        template.append_message(template.roles[0], f"<image>\n{prompt}")
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        system_header = (template.system_template.replace("{system_message}", template.system_message)
                         + template.sep)
        query = query.replace(system_header, "")
        query = query.replace(
            "<image>", "<img>" + "<IMG_CONTEXT>" * self._num_image_token * num_patches + "</img>", 1)

        tok = self._tokenizer([query], padding=True, return_tensors="pt",
                              return_offsets_mapping=True, add_special_tokens=False)
        dev = self.device
        valid = (tok["input_ids"] != self._tokenizer.pad_token_id).to(dev)
        ll = LanguageLabel(
            phrase_ids=tok["input_ids"].to(dev), phrase_valid=valid, phrase_mask=valid,
            placeholder_values=[{self._tp_token_id: np.asarray(target_points, dtype=np.float32)}],
            language_string=[query], loss_masking=None,
        )
        tp0 = torch.from_numpy(np.asarray(target_points[0], dtype=np.float32)[np.newaxis]).to(dev)
        model_input = DrivingInput(
            camera_images=processed_image.to(dev).bfloat16(),
            image_sizes=None,
            camera_intrinsics=self._K,
            camera_extrinsics=self._E,
            vehicle_speed=torch.tensor([[float(speed_mps)]], dtype=torch.float32, device=dev),
            target_point=tp0,
            prompt=ll,
            prompt_inference=ll,
        )

        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_speed_wps, pred_route, language = self._model(model_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_ms = (time.perf_counter() - t0) * 1000.0

        if pred_route is None or pred_route.numel() == 0:
            raise RuntimeError("model returned no route")
        if pred_speed_wps is None or pred_speed_wps.shape[1] < 3:
            raise RuntimeError("model returned no speed waypoints")
        lang = language[0] if isinstance(language, (list, tuple)) else str(language or "")
        return InferenceResult(
            route=pred_route[0].float().cpu().numpy().astype(np.float64),
            speed_wps=pred_speed_wps[0].float().cpu().numpy().astype(np.float64),
            language=str(lang),
            model_ms=model_ms,
        )
