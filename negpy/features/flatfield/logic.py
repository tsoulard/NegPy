import hashlib
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# Clamp so a near-black reference pixel can't blow up the image.
_GAIN_MIN = 0.25
_GAIN_MAX = 4.0

# Falloff is low-frequency, so compute the gain on a small copy, upscaled at apply time,
# and the blur kernel stays tiny.
_GAIN_WORK_SIZE = 256
# Blur sigma as a fraction of the long side: small enough to follow the light source's own
# pattern, which repeats from shot to shot. A wider blur leaves it in the corrected frame.
_BLUR_DIVISOR = 64.0

# A reference pixel below _LIT_FRACTION of the bright level (_LIT_PERCENTILE of luminance)
# is carrier, not falloff. A percentile, not the median, so a carrier that fills most of the
# frame still reads dark. Under _MIN_LIT_FRACTION lit pixels the whole frame is used.
_LIT_PERCENTILE = 95.0
_LIT_FRACTION = 0.2
_MIN_LIT_FRACTION = 0.02
# Below this blurred lit weight a pixel is too far from the lit area for the normalized blur
# to hold, and takes the value of the nearest pixel that is not.
_MIN_LIT_WEIGHT = 0.05

# Resolved gains keyed by profile id: (gain map, content token). A cached ``None`` marks a
# known-missing profile, so a broken reference does not re-hit the store every render.
# Populated lazily through the injected provider: the desktop app wires it to the on-disk
# profile store (services/assets/flatfield.py) at startup, and tests may seed this map
# directly.
GainEntry = Tuple[np.ndarray, str]
_GAIN_CACHE: Dict[str, Optional[GainEntry]] = {}
_gain_provider: Optional[Callable[[str], Optional[GainEntry]]] = None


def set_gain_provider(provider: Optional[Callable[[str], Optional[GainEntry]]]) -> None:
    """Inject the ``profile_id -> (gain, token)`` resolver and drop any cached gains."""
    global _gain_provider
    _gain_provider = provider
    _GAIN_CACHE.clear()


def invalidate_gain(profile_id: Optional[str] = None) -> None:
    """Drop a cached gain (all when None) after a profile is re-baked or deleted."""
    if profile_id is None:
        _GAIN_CACHE.clear()
    else:
        _GAIN_CACHE.pop(profile_id, None)


def _resolve(profile_id: str) -> Optional[GainEntry]:
    if not profile_id:
        return None
    if profile_id in _GAIN_CACHE:
        return _GAIN_CACHE[profile_id]
    entry = _gain_provider(profile_id) if _gain_provider is not None else None
    _GAIN_CACHE[profile_id] = entry
    return entry


def compute_gain(reference: ImageBuffer) -> np.ndarray:
    """Per-channel gain = mean(blur) / blur over the lit area, on a downsampled copy."""
    ref = reference.astype(np.float32)
    h, w = ref.shape[:2]
    scale = min(1.0, _GAIN_WORK_SIZE / max(h, w))
    if scale < 1.0:
        ref = cv2.resize(ref, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    sigma = max(ref.shape[:2]) / _BLUR_DIVISOR
    eps = 1e-4
    lit = _lit_mask(ref)
    # Normalized convolution: a dark carrier edge in the reference would otherwise bleed into
    # the blur and overcorrect the frame next to it.
    num = cv2.GaussianBlur(ref * lit[..., None], (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(lit, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blur = np.clip(num / np.clip(den, eps, None)[..., None], eps, None)
    blur = _fill_from_nearest(blur, den >= _MIN_LIT_WEIGHT)
    means = (blur * lit[..., None]).sum(axis=(0, 1)) / lit.sum()
    gain = means[None, None, :] / blur
    return np.clip(gain, _GAIN_MIN, _GAIN_MAX).astype(np.float32)


def _lit_mask(ref: np.ndarray) -> np.ndarray:
    """1 where the reference sees the light source, 0 on carrier or mask edges in frame."""
    lum = ref.mean(axis=2)
    lit = (lum > _LIT_FRACTION * np.percentile(lum, _LIT_PERCENTILE)).astype(np.uint8)
    # Drop the soft transition the downsample leaves at the carrier edge. Outside the image
    # counts as dark, so a partly lit carrier lip on the outermost row or column goes too.
    lit = cv2.erode(lit, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if lit.sum() < _MIN_LIT_FRACTION * lit.size:
        logger.warning("Flat-field: reference is almost all dark; computing the gain over the whole frame")
        return np.ones(lum.shape, np.float32)
    return lit.astype(np.float32)


def _fill_from_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace each invalid pixel with the value of the nearest valid one."""
    if valid.all() or not valid.any():
        return values
    _, labels = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    source = np.zeros(labels.max() + 1, np.int64)
    source[labels[valid]] = np.flatnonzero(valid)
    return values.reshape(-1, values.shape[2])[source[labels]].reshape(values.shape)


def gain_token(gain: np.ndarray) -> str:
    """Stable content id for a baked gain map, folded into the render source hash."""
    return hashlib.blake2b(np.ascontiguousarray(gain, dtype=np.float32).tobytes(), digest_size=8).hexdigest()


def flatfield_token(config: FlatFieldConfig) -> str:
    """Identity of the active correction, folded into the render source hash. Empty when inactive."""
    if not config.apply or not config.profile_id:
        return ""
    entry = _resolve(config.profile_id)
    if entry is None:
        return ""
    return f"|ff:{config.profile_id}:{entry[1]}"


def apply_flatfield(
    image: ImageBuffer, config: FlatFieldConfig, gain_crop: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> ImageBuffer:
    """Multiply the linear source by the reference gain map. No-op when inactive or unresolved.

    ``gain_crop`` cuts the gain to the part of the frame *image* holds, such as one half-frame;
    the gain is then resized to *image*."""
    if not config.apply or not config.profile_id:
        return image
    entry = _resolve(config.profile_id)
    if entry is None:
        return image
    gain = entry[0] if gain_crop is None else gain_crop(entry[0])
    if gain.shape[:2] != image.shape[:2]:
        gain = cv2.resize(gain, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (image * gain).astype(np.float32)
