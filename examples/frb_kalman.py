"""Kalman significance of the ASKAP example bursts.

Run from the repository root (h5py is not a package dependency):
``uv run --with h5py python examples/frb_kalman.py examples/frb171004.h5``
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy import ma

from kalman_detector.main import KalmanDetector
from kalman_detector.utils import SnrResult, sidak_logsf

# Load the frb waterfall data and detection parameters.
frb_file = sys.argv[1]  # "frb171004.h5"
frb_path = Path(frb_file)
frb_name = frb_path.stem
with h5py.File(frb_file, "r") as infile:
    dataset = infile["waterfall"]
    waterfall = np.array(dataset, dtype=float)
    snr_boxcar = dataset.attrs["snr_boxcar"]
    on_pulse_bins = dataset.attrs["on_pulse_bins"]
    pulse_width = dataset.attrs["pulse_width"]
    time_res = dataset.attrs["tsamp"]
    freqs = dataset.attrs["freqs"]

# Mask the on-pulse region and calculate the spectrum.
mask = np.ones(waterfall.shape, dtype=bool)
mask[..., range(*on_pulse_bins)] = 0
mx_on = ma.masked_array(waterfall, mask=mask, copy=True)
mx_off = ma.masked_array(waterfall, mask=~mask, copy=True)
spec = mx_on.sum(axis=-1).data
spec_std = mx_off.std(axis=-1).data * np.sqrt(pulse_width)

# Off-pulse spectra: sums over pulse_width bins, away from the burst.
guard = 50
start, stop = on_pulse_bins
off_starts = [
    t
    for t in range(0, waterfall.shape[1] - pulse_width + 1, pulse_width)
    if t + pulse_width <= start - guard or t >= stop + guard
]
off_specs = np.array([waterfall[:, t : t + pulse_width].sum(axis=1) for t in off_starts])

# Calibrate the null analytically (saddlepoint) and score the burst.
kalman = KalmanDetector(spec_std)
kalman.prepare_fits()
snr_result = SnrResult(
    frb_name,
    snr_boxcar,
    kalman.get_best_significance(spec),
    n_eff=kalman.n_eff,
)
print(snr_result)

# Robustness check: compare the analytic null with the off-pulse scores.
anchor = kalman.off_pulse_anchor(off_specs)
sig_anchored = sidak_logsf(kalman.get_best_significance(spec, off_specs), kalman.n_eff)
print(
    f"            Off-pulse check ({anchor.n_off} spectra): "
    f"std ratio {np.round(anchor.scale_ratio, 2)}, "
    f"mean offset (sigma) {np.round(anchor.mean_z, 1)}\n"
    f"            Kalman Sig: {-snr_result.sig_kalman_corrected:.1f} analytic, "
    f"{-sig_anchored:.1f} anchored",
)

# Narrowband check: best subband S/N combined with its Kalman contrast.
local = kalman.get_localized_significance(spec)
print(
    f"            Localized: channels [{local.chan_start}, {local.chan_end}), "
    f"subband S/N {local.snr_box:.1f}, corrected S/N {local.snr:.1f} "
    f"({local.n_subbands} subbands)",
)


# Plot the waterfall and profile. Pulse is centered at 0 ms.
tstart = -waterfall.shape[1] * time_res * 1e3 / 2
tstop = waterfall.shape[1] * time_res * 1e3 / 2

fig, (ax_prof, ax_dedisp) = plt.subplots(
    2,
    1,
    height_ratios=(1, 3.5),
    figsize=(6, 6.5),
    sharex=True,
)
ax_dedisp.imshow(
    waterfall,
    aspect="auto",
    extent=[tstart, tstop, freqs.min(), freqs.max()],
    cmap="magma_r",
    vmin=np.nanpercentile(waterfall, 5),
    vmax=waterfall.max(),
)
xs = np.linspace(tstart, tstop, num=waterfall.shape[1])
ax_prof.plot(xs, waterfall.mean(axis=0))
box_text = r"$\mathrm{{S/N}_{P}}$ = " + f"{snr_result.snr_box:.1f}"
kal_text = r"$\mathrm{{S/N}_{P+K}}$ = " + f"{snr_result.snr_kalman:.1f}"
ax_prof.text(0.75, 0.75, frb_name, transform=ax_prof.transAxes, fontsize=12)
ax_prof.text(0.03, 0.85, box_text, transform=ax_prof.transAxes, fontsize=12)
ax_prof.text(0.03, 0.65, kal_text, transform=ax_prof.transAxes, fontsize=12)
ax_prof.set_xlim(-60, 60)
ax_dedisp.set_xlabel("Time (ms)")
ax_dedisp.set_ylabel("Frequency (MHz)")
fig.tight_layout()
plt.savefig(frb_path.with_name(f"{frb_name}_kalman.png"), dpi=300)
