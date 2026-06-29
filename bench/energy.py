"""
Energy measurement helpers for the edge device.

Three rigor levels (see docs/piano_operativo_fase0_fase1.md):
  base:        USB-C inline power meter with logging (manual or serial readout)
  intermediate: smart plug with power logging
  rigorous:    INA219/INA260 inline DC sensor sampled by a microcontroller / 2nd Pi

This module provides a minimal interface; implement read_power() for your meter.
TODO (Claude Code, on the Pi): wire this to the actual sensor available.
"""
import time


class EnergyMeter:
    def __init__(self, sampler=None):
        # sampler() should return instantaneous power in Watts.
        self.sampler = sampler
        self._samples = []
        self._t0 = None

    def start(self):
        self._samples = []
        self._t0 = time.time()

    def sample(self):
        if self.sampler is not None:
            self._samples.append((time.time(), self.sampler()))

    def stop_joules(self):
        # trapezoidal integration of power over time -> Joules
        if len(self._samples) < 2:
            return None
        j = 0.0
        for (t0, p0), (t1, p1) in zip(self._samples, self._samples[1:]):
            j += 0.5 * (p0 + p1) * (t1 - t0)
        return j

    def joules_per_token(self, n_tokens):
        j = self.stop_joules()
        return None if j is None else j / max(n_tokens, 1)
