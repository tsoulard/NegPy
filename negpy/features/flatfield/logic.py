import hashlib
from typing import Callable, Dict, NamedTuple, Optional, Tuple

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
# A profile stores the reference at this size for Check Flat Field: twice the gain's, so dust
# and the carrier edge stay sharp on the canvas.
_CHECK_COPY_SIZE = 512

# A reference pixel below _LIT_FRACTION of the bright level (_LIT_PERCENTILE of luminance)
# is carrier, not falloff. A percentile, not the median, so a carrier that fills most of the
# frame still reads dark. Under _MIN_LIT_FRACTION lit pixels the whole frame is used.
_LIT_PERCENTILE = 95.0
_LIT_FRACTION = 0.2
_MIN_LIT_FRACTION = 0.02
# Below this blurred lit weight a pixel is too far from the lit area for the normalized blur
# to hold, and takes the value of the nearest pixel that is not.
_MIN_LIT_WEIGHT = 0.05

# The Check Flat Field view maps ±EVENNESS_RANGE about the median to black..white, and a
# profile with no stored reference shows its gain over ±GAIN_VIEW_RANGE instead. A reference
# that self-corrects to a wider spread than UNEVEN_LIMIT gets a warning at bake.
# The limit sits above the shortfall a smooth gain leaves at the frame edge on a steep
# falloff, so a good flat does not warn.
EVENNESS_RANGE = 0.10
GAIN_VIEW_RANGE = 0.15
UNEVEN_LIMIT = 0.06
# A linear decode puts the sensor's white level near 1.0; more than _CLIP_FRACTION of lit
# pixels at _CLIP_LEVEL in some channel means the reference clipped.
_CLIP_LEVEL = 0.98
_CLIP_FRACTION = 0.001

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
    ref = work_copy(reference)
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


def work_copy(reference: ImageBuffer, long_edge: int = _GAIN_WORK_SIZE) -> np.ndarray:
    """The reference at the gain's working size (or ``long_edge``), float32."""
    ref = reference.astype(np.float32)
    h, w = ref.shape[:2]
    scale = min(1.0, long_edge / max(h, w))
    if scale < 1.0:
        ref = cv2.resize(ref, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return ref


def check_copy(reference: ImageBuffer) -> np.ndarray:
    """The reference copy a profile stores for Check Flat Field."""
    return work_copy(reference, _CHECK_COPY_SIZE)


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


class Evenness(NamedTuple):
    """Spread of a frame about its own median over the lit area, as fractions."""

    low: float  # 1st percentile of the luminance deviation
    high: float  # 99th percentile of the luminance deviation
    color: float  # largest 1st/99th-percentile deviation of a channel from luminance
    clipped: bool  # lit pixels at the clip level in some channel

    @property
    def spread(self) -> float:
        return max(-self.low, self.high)


def evenness(image: ImageBuffer) -> Evenness:
    """Measure how even ``image`` is about its own lit-area median."""
    _, result = _evenness(image)
    return result


def evenness_view(image: ImageBuffer, span: float = EVENNESS_RANGE) -> Tuple[np.ndarray, Evenness]:
    """Each channel over its lit-area median, ±span mapped to black..white; carrier black."""
    ratio, result = _evenness(image)
    view = np.clip(0.5 + (ratio - 1.0) / (2.0 * span), 0.0, 1.0)
    return view.astype(np.float32), result


def check_reference(reference: ImageBuffer, gain: np.ndarray) -> Evenness:
    """Self-correct a reference's check copy with its gain and measure what is left.

    Clipping reads the full reference: the downsample averages isolated clipped pixels away.
    """
    small = check_copy(reference)
    result = evenness(self_corrected(small, gain))
    return result._replace(clipped=_is_clipped(reference.astype(np.float32)))


def self_corrected(reference: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """A reference multiplied by a gain, resized to the reference."""
    if gain.shape[:2] != reference.shape[:2]:
        gain = cv2.resize(gain, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    return reference.astype(np.float32) * gain


def _is_clipped(image: np.ndarray) -> bool:
    lit = _lit_mask(image).astype(bool)
    return float((image[lit] >= _CLIP_LEVEL).any(axis=1).mean()) > _CLIP_FRACTION


def _evenness(image: ImageBuffer) -> Tuple[np.ndarray, Evenness]:
    img = image.astype(np.float32)
    lit = _lit_mask(img).astype(bool)
    median = np.clip(np.median(img[lit], axis=0), 1e-6, None)
    ratio = img / median
    ratio[~lit] = 0.0
    lum = ratio[lit].mean(axis=1)
    low, high = np.percentile(lum - 1.0, [1.0, 99.0])
    chroma = np.percentile(ratio[lit] / np.clip(lum, 1e-6, None)[:, None] - 1.0, [1.0, 99.0], axis=0)
    clipped = float((img[lit] >= _CLIP_LEVEL).any(axis=1).mean()) > _CLIP_FRACTION
    return ratio, Evenness(float(low), float(high), float(np.abs(chroma).max()), clipped)


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
