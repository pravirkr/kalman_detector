import sys

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy import ma

from kalman_detector.main import KalmanDetector
from kalman_detector.utils import SnrResult


def kalman_calc(
    frb_name: str,
    spec: np.ndarray,
    spec_std: np.ndarray,
    snr_box: float,
    ntrials: int = 10000,
) -> SnrResult:
    kalman = KalmanDetector(spec_std)
    kalman.prepare_fits(ntrials=ntrials)
    sig_kalman = kalman.get_best_significance(spec)
    return SnrResult(frb_name, snr_box, sig_kalman)

# Load the frb waterfall data and detection parameters.
frb_file = sys.argv[1] # "frb171004.h5"
frb_name = frb_file.split(".")[0]
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

# Calculate the Kalman significance and the combined S/N.
snr_result = kalman_calc(frb_name, spec, spec_std, snr_boxcar)
print(snr_result)


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
box_text = "$\mathrm{{S/N}_{P}}$ = " + f"{snr_result.snr_box:.1f}"
kal_text = "$\mathrm{{S/N}_{P+K}}$ = " + f"{snr_result.snr_kalman:.1f}"
ax_prof.text(0.75, 0.75, frb_name, transform=ax_prof.transAxes, fontsize=12)
ax_prof.text(0.03, 0.85, box_text, transform=ax_prof.transAxes, fontsize=12)
ax_prof.text(0.03, 0.65, kal_text, transform=ax_prof.transAxes, fontsize=12)
ax_prof.set_xlim(-60, 60)
ax_dedisp.set_xlabel("Time (ms)")
ax_dedisp.set_ylabel("Frequency (MHz)")
fig.tight_layout()
plt.savefig(f"{frb_name}_kalman.png", dpi=300)
