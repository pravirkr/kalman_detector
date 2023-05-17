# kalman_detector

[![GitHub CI](https://github.com/pravirkr/kalman_detector/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pravirkr/kalman_detector/actions/workflows/ci.yml)
[![codecov](https://app.codecov.io/gh/pravirkr/kalman_detector/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pravirkr/kalman_detector)
[![License](https://img.shields.io/github/license/pravirkr/kalman_detector)](https://github.com/pravirkr/kalman_detector/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python Implementation of the Kalman filter detector for detecting smoothly variying signals buried in gaussian noise (like Fast Radio Bursts). 

Score is designed to receive X(f), a sequence of "amplitudes" (where f is an arbitrary indexed parameter) and decide between:

```
H0: X(f) = N(f)         Pure gaussian noise
H1: X(f) = A(f) + N(f)  A(f) is a smooth gaussian process with the smoothness parameter unknown. 
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
