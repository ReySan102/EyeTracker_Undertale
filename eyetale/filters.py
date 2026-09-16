"""
Signal conditioning: the One Euro filter and the Schmitt trigger.

Both are tiny, pure, and used everywhere a noisy gaze signal has to become a
stable decision.

* OneEuroFilter  - smooths a jittery coordinate with very little lag
                   (Casiez, Roussel & Vogel, CHI 2012).
* SchmittTrigger - turns a continuous error value into -1 / 0 / +1 with
                   hysteresis, so a key does not flicker when the value sits
                   on a boundary.

Run:  python -m eyetale.filters
"""
from __future__ import annotations

import math


class LowPassFilter:
    def __init__(self) -> None:
        self._y: float | None = None

    def reset(self) -> None:
        self._y = None

    @property
    def last(self) -> float | None:
        return self._y

    def apply(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y

        return self._y


class OneEuroFilter:
    """
    Adaptive low-pass filter for a single coordinate.

    The cutoff frequency rises with the speed of the signal: when the eye is
    still, jitter is smoothed hard; during a fast movement the filter opens up
    so the output does not lag behind.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = LowPassFilter()
        self._dx = LowPassFilter()
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)

        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self._t_prev is None or t <= self._t_prev:
            dt = 1.0 / 60.0
        else:
            dt = t - self._t_prev

        self._t_prev = t

        prev = self._x.last
        dx = 0.0 if prev is None else (x - prev) / dt
        dx_hat = self._dx.apply(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        return self._x.apply(x, self._alpha(cutoff, dt))


class OneEuroFilter2D:
    """Two independent One Euro filters, one per axis."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def filter(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx.filter(x, t), self.fy.filter(y, t)


class SchmittTrigger:
    """
    Three-state comparator with hysteresis.

    state 0  -> +1 when value >  enter
    state 0  -> -1 when value < -enter
    state +1 ->  0 when value <  exit      (exit < enter, that gap is the hysteresis)
    state -1 ->  0 when value > -exit
    A direct flip from +1 to -1 happens only when the value crosses -enter.
    """

    def __init__(self, enter: float, exit: float) -> None:
        if exit > enter:
            raise ValueError("exit threshold must be <= enter threshold")

        self.enter = float(enter)
        self.exit = float(exit)
        self.state = 0

    def reset(self) -> None:
        self.state = 0

    def set_thresholds(self, enter: float, exit: float) -> None:
        self.enter, self.exit = float(enter), float(exit)

    def update(self, value: float) -> int:
        if self.state == 0:
            if value > self.enter:
                self.state = 1
            elif value < -self.enter:
                self.state = -1
        elif self.state == 1:
            if value < -self.enter:
                self.state = -1
            elif value < self.exit:
                self.state = 0
        else:  # state == -1
            if value > self.enter:
                self.state = 1
            elif value > -self.exit:
                self.state = 0

        return self.state


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import random

    # One Euro: a constant signal with noise must come out much less noisy.
    random.seed(1)
    f = OneEuroFilter(min_cutoff = 1.0, beta = 0.0)
    outs = [f.filter(100.0 + random.uniform(-10, 10), i / 60.0) for i in range(300)]
    tail = outs[120:]
    spread = max(tail) - min(tail)
    assert spread < 8.0, f"filter did not smooth: spread={spread}"

    # One Euro: a step must be followed within a reasonable time (no huge lag).
    f = OneEuroFilter(min_cutoff = 1.0, beta = 0.05)
    y = 0.0

    for i in range(60):
        y = f.filter(0.0 if i < 10 else 500.0, i / 60.0)

    assert y > 450.0, f"filter lags too much after a step: {y}"

    # Schmitt trigger: hysteresis keeps state between exit and enter.
    s = SchmittTrigger(enter = 10, exit = 4)
    assert s.update(5) == 0          # Below enter, still idle
    assert s.update(11) == 1         # Crossed enter
    assert s.update(6) == 1          # Inside the hysteresis band, keep holding
    assert s.update(3) == 0          # Dropped below exit, release
    assert s.update(-11) == -1
    assert s.update(-5) == -1
    assert s.update(11) == 1         # Direct flip
    assert s.update(-11) == -1

    try:
        SchmittTrigger(enter = 1, exit = 2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    print("filters: OK")


if __name__ == "__main__":
    _self_test()
