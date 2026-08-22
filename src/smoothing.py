# -*- coding: utf-8 -*-
"""Kalman filter and One-Euro filter for fingertip smoothing."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import defaultdict


@dataclass
class KalmanState:
    x: np.ndarray
    P: np.ndarray
    initialized: bool = False
    last_timestamp: float = 0.0


class FingertipKalmanFilter:
    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
        dt: float = 1/30.0,
    ):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.filters: Dict[str, KalmanState] = defaultdict(lambda: None)

        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        self.Q = np.eye(6) * process_noise
        self.Q[0, 0] = self.Q[1, 1] = self.Q[2, 2] = process_noise * dt**2 / 3
        self.Q[3, 3] = self.Q[4, 4] = self.Q[5, 5] = process_noise

        self.R = np.eye(3) * measurement_noise
        self.R[2, 2] = measurement_noise * 10

    def update(self, finger: str, x: float, y: float, z: float, timestamp: float) -> tuple:
        state = self.filters[finger]

        if state is None or not state.initialized:
            state = KalmanState(
                x=np.array([[x], [y], [z], [0], [0], [0]], dtype=float),
                P=np.eye(6) * 0.01,
                initialized=True,
                last_timestamp=timestamp,
            )
            self.filters[finger] = state
            return x, y, z

        dt = timestamp - state.last_timestamp
        if dt <= 0 or dt > 0.5:
            dt = self.dt
        state.last_timestamp = timestamp

        F = self.F.copy()
        F[0, 3] = F[1, 4] = F[2, 5] = dt

        x_pred = F @ state.x
        P_pred = F @ state.P @ F.T + self.Q

        z_meas = np.array([[x], [y], [z]], dtype=float)
        y_residual = z_meas - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        state.x = x_pred + K @ y_residual
        state.P = (np.eye(6) - K @ self.H) @ P_pred

        return float(state.x[0, 0]), float(state.x[1, 0]), float(state.x[2, 0])

    def get_velocity(self, finger: str) -> tuple:
        state = self.filters.get(finger)
        if state and state.initialized:
            return float(state.x[3, 0]), float(state.x[4, 0]), float(state.x[5, 0])
        return 0.0, 0.0, 0.0

    def reset(self, finger: str):
        if finger in self.filters:
            self.filters[finger].initialized = False


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        dx = (x - self.x_prev) / dt
        edx = self._exponential_smoothing(dx, self.d_cutoff, dt)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        x_hat = self._exponential_smoothing(x, cutoff, dt)

        self.x_prev = x_hat
        self.dx_prev = edx
        self.t_prev = t
        return x_hat

    def _exponential_smoothing(self, x, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)
        return alpha * x + (1 - alpha) * self.x_prev


class MultiAxisOneEuroFilter:
    def __init__(self, **kwargs):
        self.filters = {}
        self.kwargs = kwargs

    def update(self, finger: str, x: float, y: float, z: float, timestamp: float) -> tuple:
        if finger not in self.filters:
            self.filters[finger] = {
                'x': OneEuroFilter(**self.kwargs),
                'y': OneEuroFilter(**self.kwargs),
                'z': OneEuroFilter(**self.kwargs),
            }
        f = self.filters[finger]
        return (
            f['x'](x, timestamp),
            f['y'](y, timestamp),
            f['z'](z, timestamp),
        )