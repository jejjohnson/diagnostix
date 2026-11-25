"""The mother wavelets of Torrence and Compo (1998), and their constants.

Torrence, C., Compo, G. P. (1998). A practical guide to wavelet analysis.
Bulletin of the American Meteorological Society 79(1), 61–78.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from jaxtyping import Float

from diagnostix.exceptions import ValidationError


__all__ = ["Mother"]

#: The parameter of each mother: ω₀, the order, and the derivative.
DEFAULTS = {"morlet": 6.0, "paul": 4.0, "dog": 2.0}
#: Below this ω₀ a Morlet wavelet has a mean, and its inverse no constant.
SMALLEST_MORLET = 5.0


@dataclasses.dataclass(frozen=True)
class Mother:
    """A mother wavelet: ``"morlet"``, ``"paul"`` or ``"dog"``.

    ``"dog"`` is a derivative of a Gaussian, the Mexican hat for the second. It
    is real and the others are complex, with no negative frequencies, and real
    and positive in Fourier space: Torrence and Compo's Paul of order m is iᵐ
    times this one in space, and the same for their order of 4.
    """

    name: str
    param: float

    @classmethod
    def of(cls, name: str, param: float | None) -> Mother:
        """The mother of a name, with its own parameter if none is given.

        Raises:
            ValidationError: If there is no such mother, or no such parameter.
        """
        if name not in DEFAULTS:
            msg = f"'mother' must be one of {', '.join(DEFAULTS)}, got {name!r}"
            raise ValidationError(msg)
        value = DEFAULTS[name] if param is None else float(param)
        whole = name == "morlet" or value.is_integer()
        if not (math.isfinite(value) and value > 0.0 and whole):
            kind = "number" if name == "morlet" else "whole number"
            msg = f"'param' of {name!r} must be a positive {kind}, got {param!r}"
            raise ValidationError(msg)
        return cls(name, value)

    @property
    def degrees_of_freedom(self) -> int:
        """2 for a complex wavelet, whose power is a sum of two squares; else 1."""
        return 1 if self.name == "dog" else 2

    @property
    def period(self) -> float:
        """The Fourier period of a scale of 1."""
        if self.name == "morlet":
            return 4.0 * math.pi / (self.param + math.sqrt(2.0 + self.param**2))
        if self.name == "paul":
            return 4.0 * math.pi / (2.0 * self.param + 1.0)
        return 2.0 * math.pi / math.sqrt(self.param + 0.5)

    @property
    def reach(self) -> float:
        """The e-folding distance of an edge at a scale of 1."""
        return math.sqrt(0.5) if self.name == "paul" else math.sqrt(2.0)

    @property
    def phase(self) -> complex:
        """The constant phase of the conjugate of ψ̂₀: ±1 or ±i for ``"dog"``."""
        return -(1j ** (int(self.param) % 4)) if self.name == "dog" else 1.0 + 0.0j

    @property
    def is_real_in_fourier(self) -> bool:
        """Whether ψ̂₀ is real, so that the phase is ±1."""
        return self.phase.imag == 0.0

    def fourier(self, eta: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
        """The conjugate of ψ̂₀(η) without its `phase`: real."""
        if self.name == "dog":
            m = self.param
            norm = math.sqrt(1.0 / math.gamma(m + 0.5))
            return norm * eta**m * np.exp(-0.5 * eta**2)
        positive = np.where(eta > 0.0, eta, 0.0)
        if self.name == "morlet":
            values = math.pi**-0.25 * np.exp(-0.5 * (positive - self.param) ** 2)
        else:
            m = self.param
            norm = 2.0**m / math.sqrt(m * math.gamma(2.0 * m))
            values = norm * positive**m * np.exp(-positive)
        return np.where(eta > 0.0, values, 0.0)

    def inverse_constant(self) -> float:
        """C, by which a sum over scales is divided to give the signal back.

        C = (√(2π) / (2 ln 2)) ∫₀^∞ ψ̂₀(η)/η dη for a complex mother, and twice
        that for a real one, whose negative frequencies count too.

        Raises:
            ValidationError: If the mother has no such constant.
        """
        m = self.param
        if not self.is_real_in_fourier or (
            self.name == "morlet" and m < SMALLEST_MORLET
        ):
            msg = (
                f"a {self.name!r} of param {m:g} has no inverse: 'dog' needs an "
                f"even one, and 'morlet' at least {SMALLEST_MORLET:g}"
            )
            raise ValidationError(msg)
        if self.name == "morlet":
            # the integrand is below 1e-5 of its largest where it is cut
            eta = np.linspace(max(m - 12.0, 1e-3), m + 12.0, 40001)
            integral = float(np.trapezoid(self.fourier(eta) / eta, eta))
        elif self.name == "paul":
            # ∫ η^(m−1) e^(−η) dη = Γ(m)
            norm = 2.0**m / math.sqrt(m * math.gamma(2.0 * m))
            integral = norm * math.gamma(m)
        else:
            # ∫ η^(m−1) e^(−η²/2) dη = 2^(m/2 − 1) Γ(m/2), and both halves
            norm = math.sqrt(1.0 / math.gamma(m + 0.5))
            integral = 2.0 * norm * 2.0 ** (0.5 * m - 1.0) * math.gamma(0.5 * m)
        return self.phase.real * integral * math.sqrt(2.0 * math.pi) / math.log(4.0)
