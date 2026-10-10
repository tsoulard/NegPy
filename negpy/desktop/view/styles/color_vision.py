"""Color vision: one Preferences choice, read by every view that tells things apart by hue alone.

A palette names roles, not hues, so a view asks for its role and gets colors that vision keeps.
"""

from dataclasses import dataclass

from negpy.desktop.view.styles.theme import THEME


@dataclass(frozen=True)
class VisionPalette:
    key: str
    label: str
    pair: tuple[str, str]  # two kinds of mark that must read apart over any image
    dodge_burn: tuple[str, str]  # Dodge & Burn masks: warm for dodge, cool for burn


# Standard's pair is neon: a mark has to read over any film. The color-blind colors are
# Okabe–Ito (Color Universal Design) colors on an axis that vision keeps. A pair keeps clear of
# the amber exclusion band on the same overlay, and every color reads as text on the dark panel.
PALETTES: tuple[VisionPalette, ...] = (
    VisionPalette("standard", "Standard", ("#39FF14", "#FF00FF"), (THEME.dodge, THEME.burn)),
    VisionPalette("protan_deutan", "Protanopia / deuteranopia (red-green)", ("#56B4E9", "#D55E00"), ("#E69F00", "#56B4E9")),
    VisionPalette("tritan", "Tritanopia (blue-yellow)", ("#D55E00", "#009E73"), ("#D55E00", "#009E73")),
    VisionPalette("achromat", "Achromatopsia (no color)", ("#FFFFFF", "#8C8C8C"), ("#FFFFFF", "#8C8C8C")),
)


def palette_for(key: str) -> VisionPalette:
    return next((p for p in PALETTES if p.key == key), PALETTES[0])
