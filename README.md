# kalman_detector

[![GitHub CI](https://github.com/pravirkr/kalman_detector/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pravirkr/kalman_detector/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pravirkr/kalman_detector/branch/main/graph/badge.svg)](https://codecov.io/gh/pravirkr/kalman_detector)
[![License](https://img.shields.io/github/license/pravirkr/kalman_detector)](https://github.com/pravirkr/kalman_detector/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python Implementation of the Kalman filter detector for detecting smoothly variying signals buried in gaussian noise (like Fast Radio Bursts).

The detection statistic is designed to receive I(f), a sequence of observed "amplitudes" (where f is an arbitrary indexed parameter) and decide between:

```
H0: I(f) = N(f)         Pure gaussian noise
H1: I(f) = A(f) + N(f)  A(f) is a smooth gaussian process with the smoothness parameter unknown. 
```

## Installation

The quickest way to install the package is to use [pip](https://pip.pypa.io):

```bash
pip install -U git+https://github.com/pravirkr/kalman_detector
```

## Usage

```python
from kalman_detector.main import KalmanDetector

kalman = KalmanDetector(spectrum_std)
kalman.prepare_fits(ntrials=10000)
kalman.get_significance(spectrum)
```

### Efficiency

```python
python -m kalman_detector.efficiency
```

## Citation

Please cite [Kumar, Zackay & Law (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJ...960..128K) if you find this code useful in your research.
The BibTeX entry for the paper is:

```bash
@ARTICLE{2024ApJ...960..128K,
       author = {{Kumar}, Pravir and {Zackay}, Barak and {Law}, Casey J.},
        title = "{Detecting Fast Radio Bursts with Spectral Structure Using the Continuous Forward Algorithm}",
      journal = {\apj},
     keywords = {Radio astronomy, Radio transient sources, Astronomy data analysis, Astrostatistics techniques, Interstellar scintillation, 1338, 2008, 1858, 1886, 855, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = jan,
       volume = {960},
       number = {2},
          eid = {128},
        pages = {128},
          doi = {10.3847/1538-4357/ad0964},
archivePrefix = {arXiv},
       eprint = {2306.07914},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...960..128K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
