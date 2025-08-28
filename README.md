# xentropy

Entropy measures for time series analysis with NumPy and Xarray.

Implements:
- Approximate Entropy (ApEn, Pincus 1991)
- Sample Entropy (SampEn, Richman and Moorman 2000)
- Cross-ApEn / Cross-SampEn (Costa et al. 2002)
- Multiscale Entropy (MSE) (Costa et al. 2002)


## Installation

```bash
pip install git+https://github.com/senclimate/xentropy.git
```

## Quick Start

```python
import xarray as xr
import numpy as np
from xentropy import xentropy

# create sample data
time = np.arange(1000)
data = np.sin(0.1 * time) + 0.1*np.random.randn(1000)
da = xr.DataArray(data, dims=["time"])

xe = xentropy(dim='time')
SampEn = xe.SampEn(da)
print(SampEn.values)
```

## Applications
- ENSO reguliarity in observation, AWI-CM3 and CMIP6 models (Fig. 2 in Stuecker et al. 2025)

## Acknowledgement

If you use the xentropy code in your published work, please kindly cite:

Stuecker, M. F., Zhao, S., Timmermann, A., Ghosh, R., Semmler, T., Lee, S.-S., et al. (2025). Global climate mode resonance due to rapidly intensifying El Niño–Southern Oscillation. **Nature Communications**. in revision. 


## References

- Pincus, S. M. (1991). Approximate entropy as a measure of system complexity. PNAS, 88(6), 2297–2301.
- Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. AJP, 278(6), H2039–H2049.
- Costa, M., Goldberger, A. L., & Peng, C. K. (2002). Multiscale entropy analysis of complex physiologic time series. PRL, 89(6), 068102.
- Pincus, S. M., & Singer, B. H. (1996). Randomness and degrees of irregularity. PNAS, 93(5), 2083–2088.

