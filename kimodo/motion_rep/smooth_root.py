# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smooth root trajectory: ADMM-based smoother with margin constraints and get_smooth_root_pos helper."""

import math
import os

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

from kimodo.tools import ensure_batched


class TrajectorySmoother:
    """Modify trajectories to hit target values while respecting soft constraints.

    This smoother keeps the trajectory close to the original positions while minimizing
    accelerations. Targets are enforced at specified frames via soft constraints.
    """

    def __init__(
        self,
        margins,
        pos_weight=0.0,
        loop=False,
        admm_iters=100,
        alpha_overrelax=1.0,
        circle_project=False,
    ):
        """Initialize the TrajectorySmoother.

        Args:
            margins: Array of margin values for each frame.
                    margins[i] < 0: unconstrained
                    margins[i] == 0: pinned on this frame
                    margins[i] > 0: can deviate within the margin
            pos_weight: Weight for position preservation
            loop: Whether the trajectory should loop
            admm_iters: Number of ADMM iterations
        """
        self.pos_weight = pos_weight
        self.admm_iters = admm_iters
        self.alpha_overrelax = alpha_overrelax
        self.circle_project = circle_project
        N = len(margins)

        # Store margin information as numpy arrays
        self.margin_vals = margins

        # Build acceleration matrix A
        a_data = []
        a_rows = []
        a_cols = []

        for i in range(1, N - 1):
            scale = 1.0
            a_data.extend([-scale, 2.0 * scale, -scale])
            a_rows.extend([i, i, i])
            a_cols.extend([i - 1, i, i + 1])

        if loop:
            # Add periodic accelerations
            scale = 1.0
            a_data.extend([-scale, 2.0 * scale, -scale])
            a_rows.extend([0, 0, 0])
            a_cols.extend([N - 1, 0, 1])

            scale = 1.0
            a_data.extend([-scale, 2.0 * scale, -scale])
            a_rows.extend([N - 1, N - 1, N - 1])
            a_cols.extend([N - 2, N - 1, 0])

        A = sparse.csr_matrix((a_data, (a_rows, a_cols)), shape=(N, N))

        # Build identity matrix
        identity_matrix = sparse.eye(N)

        # Build system matrix M
        M = pos_weight * identity_matrix + A.T @ A

        # Calculate ADMM step size
        diag_max = max(abs(M.diagonal()))
        self.admm_stepsize = 0.25 * np.sqrt(diag_max)

        M = M + self.admm_stepsize * identity_matrix
        self.system_lu = splu(M.tocsc())

    def smooth(self, targets, x0):
        """Interpolate between reference positions while satisfying constraints.

        Args:
            observations: Target positions for constrained frames (numpy array)
            ref_positions: Reference positions defining original shape
                         (numpy array)

        Returns:
            Interpolated positions (numpy array)
        """
        x_target = targets.copy()
        x = x0.copy()
        z = np.zeros_like(x)
        u = np.zeros_like(x)

        for _ in range(self.admm_iters):
            self.z_update(z, x, x_target, u)
            self.u_update(u, x, z)
            self.x_update(x, z, u, x_target)

        return x

    def x_update(self, x, z, u, x_t):
        """Update x in the ADMM iteration."""

        # x = (wp * I + A^T A + p I)^-1 (wp * x_orig + p (z - u))
        r = self.pos_weight * x_t + self.admm_stepsize * (z - u)
        x[:] = self.system_lu.solve(r)

    def z_update(self, z, x, z_t, u):
        """Update z in the ADMM iteration using vectorized operations."""
        # Compute the difference from target for all margin locations at once
        z[:] = x + u - z_t

        # Check if we need to project back to margin
        # Avoid np.linalg.norm here: in some mixed NumPy installs this can
        # intermittently crash with a bogus reduce() argument error.
        z_diff_norms = np.sqrt(np.sum(z * z, axis=1))
        mask = z_diff_norms > self.margin_vals
        # Avoid relying on np.any() internals from a potentially mixed NumPy install;
        # ndarray.any() is equivalent here and more robust in affected environments.
        if bool(mask.any()):
            scale_factors = self.margin_vals[mask] / z_diff_norms[mask]
            z[mask] *= scale_factors[:, np.newaxis]

        # Add back the target
        z[:] += z_t

        if self.circle_project:
            z_norms = np.sqrt(np.sum(z * z, axis=1, keepdims=True))
            z[:] = z / (z_norms + 1.0e-6)

    def u_update(self, u, x, z):
        """Update u in the ADMM iteration using vectorized operations."""
        u[:] += self.alpha_overrelax * (x - z)


def smooth_signal(x, margins, pos_weight=0, alpha_overrelax=1.8, admm_iters=500, circle_project=False):
    """Multigrid trajectory smoothing with margin constraints.

    Args:
        x: Input trajectory ``[T, D]`` as a NumPy array.
        margins: Allowed radius around each target frame ``[T]``.
        pos_weight: Weight for staying close to the original signal.
        alpha_overrelax: ADMM over-relaxation coefficient.
        admm_iters: ADMM iterations per multigrid level.
        circle_project: If ``True``, project each vector to the unit sphere.

    Returns:
        Smoothed trajectory of shape ``[T, D]``.
    """
    # Force numeric arrays to avoid rare object-dtype propagation issues.
    x = np.asarray(x, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"smooth_signal expects x with shape [T, D], got shape {x.shape}.")
    if margins.ndim != 1 or margins.shape[0] != x.shape[0]:
        raise ValueError(
            f"smooth_signal expects margins with shape [T], got {margins.shape} for x.shape={x.shape}."
        )
    if x.shape[0] == 0:
        return x.copy()
    if x.shape[0] < 5:
        # ADMM system can become singular for very short trajectories.
        return x.copy()
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, copy=False)
    if not np.all(np.isfinite(margins)):
        margins = np.nan_to_num(margins, copy=False, nan=0.06, posinf=0.06, neginf=0.0)

    x_smoothed = x.copy()
    x_smoothed[:] = x.mean(axis=0, keepdims=True)

    # smooth the signal, multigrid style by starting out coarse,
    # doubling the resolution and repeating until we're at the full
    # resolution, using the previous result as the initial guess.
    levels = int(math.floor(math.log2(len(x))))
    levels = max(levels - 4, 1)

    stepsize = 2**levels
    while True:
        # smooth signals at this level:
        num_steps = len(x_smoothed[::stepsize])
        smoother = TrajectorySmoother(
            margins=margins[::stepsize],
            pos_weight=pos_weight,
            alpha_overrelax=alpha_overrelax,
            admm_iters=admm_iters,
            circle_project=circle_project,
        )
        try:
            x_smoothed[::stepsize] = smoother.smooth(x[::stepsize], x_smoothed[::stepsize])
        except Exception:
            # Keep training robust on rare smoother failures by falling back to
            # the unsmoothed signal at the current multigrid level.
            x_smoothed[::stepsize] = x[::stepsize]

        # interpolate to next level:
        next_stepsize = stepsize // 2
        num_interleaved = len(x_smoothed[next_stepsize::stepsize])
        if num_interleaved == num_steps:
            # linearly extrapolate the last value if we have to:
            x_smoothed[next_stepsize::stepsize][-1] = (
                x_smoothed[::stepsize][-1] + (x_smoothed[::stepsize][-1] - x_smoothed[::stepsize][-2]) / 2
            )
            num_interleaved = num_interleaved - 1

        # linearly interpolate the remaining values:
        x_smoothed[next_stepsize::stepsize][:num_interleaved] = (
            x_smoothed[::stepsize][:-1] + x_smoothed[::stepsize][1:]
        ) / 2

        if stepsize == 1:
            break

        stepsize //= 2

    return x_smoothed


@ensure_batched(hip_translations=3)
def get_smooth_root_pos(hip_translations):
    """Smooth root trajectory in the ground plane while preserving height.

    Args:
        hip_translations: Root translations ``[B, T, 3]``.

    Returns:
        Smoothed root translations ``[B, T, 3]`` where ``x/z`` are smoothed and
        ``y`` remains unchanged.
    """
    if os.environ.get("KIMODO_DISABLE_ROOT_SMOOTH", "0") == "1":
        # Escape hatch for environments where scipy/numpy sparse solves may
        # intermittently crash with native floating-point signals.
        return hip_translations

    root_translations_xz = hip_translations[..., [0, 2]]
    root_translations_y = hip_translations[..., [1]]

    batch_size, nframes = root_translations_xz.shape[:2]
    margins = np.full(root_translations_xz.shape[1], 0.06)

    root_translations_smoothed_xz = []
    for batch in range(batch_size):
        root_translations_smoothed_xz.append(
            smooth_signal(root_translations_xz[batch].detach().cpu().numpy(), margins)[None]
        )

    root_translations_smoothed_xz = torch.tensor(np.concatenate(root_translations_smoothed_xz))

    root_translations = torch.cat(
        [
            root_translations_smoothed_xz.to(root_translations_y.device),
            root_translations_y,
        ],
        dim=-1,
    )[..., [0, 2, 1]]

    return root_translations
