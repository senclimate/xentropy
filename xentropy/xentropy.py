import numpy as np
import xarray as xr

class xentropy:
    """
    Entropy measures for time series analysis.
    
    Implements:
    - Approximate Entropy (ApEn)
    - Sample Entropy (SampEn)
    - Cross-Approximate Entropy (Cross-ApEn)
    - Cross-Sample Entropy (Cross-SampEn)
    - Multiscale Entropy (MSE)

    References
    ----------
    Pincus, S. M. (1991). "Approximate entropy as a measure of system complexity."
        Proceedings of the National Academy of Sciences, 88(6), 2297–2301.
    Richman, J. S., & Moorman, J. R. (2000). "Physiological time-series analysis using 
        approximate entropy and sample entropy." American Journal of Physiology, 278(6), H2039–H2049.
    Costa, M., Goldberger, A. L., & Peng, C. K. (2002). "Multiscale entropy analysis 
        of complex physiologic time series." Physical Review Letters, 89(6), 068102.

    Creator
    -------
        Sen Zhao <zhaos@hawaii.edu>,
        Last update 08/28/2025
        
    """

    def __init__(self, dim="time"):
        """
        Parameters
        ----------
        dim : str
            Dimension name for the time axis when applying to xarray objects.
        """
        self.dim = dim

    # ----------------------------
    # Helper methods
    # ----------------------------
    @staticmethod
    def _embed(x, m, tau=1):
        """
        Time-delay embedding of a 1D signal.
        
        Parameters
        ----------
        x : array_like
            Input time series.
        m : int
            Embedding dimension.
        tau : int, optional
            Time delay between points (default is 1).
        
        Returns
        -------
        ndarray
            Embedded vectors of shape (N - (m-1)*tau, m).
        """
        N = len(x)
        if N - (m - 1) * tau <= 0:
            return np.array([])
        return np.array([x[i:i + m*tau:tau] for i in range(N - (m-1)*tau)])

    @staticmethod
    def _coarse_grain(x, scale):
        """
        Coarse-graining for multiscale entropy.
        
        Parameters
        ----------
        x : array_like
            Input time series.
        scale : int
            Scale factor for coarse-graining.
        
        Returns
        -------
        ndarray
            Coarse-grained time series.
        """
        N = len(x)
        if N < scale:
            return np.array([])
        new_len = N // scale
        return np.mean(x[:new_len * scale].reshape(new_len, scale), axis=1)

    # ----------------------------
    # Approximate Entropy (ApEn)
    # ----------------------------
    @staticmethod
    def _apen_1d(x, m, r, tau, exclude_self_match=False):
        """
        Compute Approximate Entropy (ApEn) of a 1D signal.

        Parameters
        ----------
        x : array_like
            Input time series.
        m : int
            Embedding dimension.
        r : float or None
            Tolerance (if None, set to 0.2 * std(x)).
        tau : int
            Time delay.
        exclude_self_match : bool, optional
            Whether to exclude self-matches (default False).
        
        Returns
        -------
        float
            Approximate Entropy value.
        """
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        N = len(x)
        if N <= (m+1)*tau:
            return np.nan

        r_val = float(r) if r is not None else 0.2 * np.std(x)

        def phi(m_dim):
            Y = xentropy._embed(x, m_dim, tau)
            n = Y.shape[0]
            if n <= 1:
                return np.nan
            dist = np.max(np.abs(Y[:, None, :] - Y[None, :, :]), axis=2)
            if exclude_self_match:
                np.fill_diagonal(dist, np.inf)
                C = np.sum(dist <= r_val, axis=1) / (n - 1)
            else:
                C = np.sum(dist <= r_val, axis=1) / n
            return np.mean(np.log(C + 1e-12))

        return phi(m) - phi(m+1)

    # ----------------------------
    # Sample Entropy (SampEn)
    # ----------------------------
    @staticmethod
    def _sampen_1d(x, m, r, tau):
        """
        Compute Sample Entropy (SampEn) of a 1D signal.
        
        Parameters
        ----------
        x : array_like
            Input time series.
        m : int
            Embedding dimension.
        r : float or None
            Tolerance (if None, set to 0.2 * std(x)).
        tau : int
            Time delay.
        
        Returns
        -------
        float
            Sample Entropy value.
        """
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        N = len(x)
        if N <= (m+1)*tau:
            return np.nan

        r_val = float(r) if r is not None else 0.2 * np.std(x)

        def count_m(md):
            Y = xentropy._embed(x, md, tau)
            n = Y.shape[0]
            if n <= 1:
                return 0
            dist = np.max(np.abs(Y[:, None, :] - Y[None, :, :]), axis=2)
            i_upper = np.triu_indices(n, k=1)
            return np.sum(dist[i_upper] <= r_val)

        B = count_m(m)
        A = count_m(m+1)
        if B == 0:
            return np.inf
        if A == 0:
            return -np.log(1.0 / (B + 1e-12))
        return -np.log(A / B)

    # ----------------------------
    # Cross-ApEn and Cross-SampEn
    # ----------------------------
    @staticmethod
    def _cross_apen_1d(x, y, m, r, tau, exclude_self_match=False):
        """
        Compute Cross-Approximate Entropy (Cross-ApEn) between two signals.
        
        Parameters
        ----------
        x, y : array_like
            Input time series (must have same length).
        m : int
            Embedding dimension.
        r : float or None
            Tolerance (if None, uses average std of x and y).
        tau : int
            Time delay.
        exclude_self_match : bool
            Whether to exclude self matches if x is y.
        
        Returns
        -------
        float
            Cross-ApEn value.
        """
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        y = np.asarray(y, float); y = y[~np.isnan(y)]
        if len(x) != len(y):
            N = min(len(x), len(y))
            x, y = x[:N], y[:N]
        N = len(x)
        if N <= (m+1)*tau:
            return np.nan

        sigma = (np.std(x) + np.std(y)) / 2
        r_val = float(r) if r is not None else 0.2 * sigma

        def phi(m_dim):
            X = xentropy._embed(x, m_dim, tau)
            Y = xentropy._embed(y, m_dim, tau)
            n = X.shape[0]
            if n <= 1:
                return np.nan
            dist = np.max(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
            if exclude_self_match and x is y:
                np.fill_diagonal(dist, np.inf)
                C = np.sum(dist <= r_val, axis=1) / (n - 1)
            else:
                C = np.sum(dist <= r_val, axis=1) / n
            return np.mean(np.log(C + 1e-12))

        return phi(m) - phi(m+1)

    @staticmethod
    def _cross_sampen_1d(x, y, m, r, tau):
        """
        Compute Cross-Sample Entropy (Cross-SampEn) between two signals.
        
        Parameters
        ----------
        x, y : array_like
            Input time series (must have same length).
        m : int
            Embedding dimension.
        r : float or None
            Tolerance (if None, uses average std of x and y).
        tau : int
            Time delay.
        
        Returns
        -------
        float
            Cross-SampEn value.
        """
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        y = np.asarray(y, float); y = y[~np.isnan(y)]
        if len(x) != len(y):
            N = min(len(x), len(y))
            x, y = x[:N], y[:N]
        N = len(x)
        if N <= (m+1)*tau:
            return np.nan

        sigma = (np.std(x) + np.std(y)) / 2
        r_val = float(r) if r is not None else 0.2 * sigma

        def count_m(md):
            X = xentropy._embed(x, md, tau)
            Y = xentropy._embed(y, md, tau)
            dist = np.max(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
            return np.sum(dist <= r_val)

        B = count_m(m)
        A = count_m(m+1)
        if B == 0:
            return np.inf
        if A == 0:
            return -np.log(1.0 / (B + 1e-12))
        return -np.log(A / B)

    # ----------------------------
    # Public interface for ApEn and SampEn
    # ----------------------------
    def ApEn(self, da, m=2, r=None, tau=1, exclude_self_match=False):
        """Compute Approximate Entropy over xarray objects."""
        return xr.apply_ufunc(
            self._apen_1d, da,
            kwargs={"m": m, "r": r, "tau": tau, "exclude_self_match": exclude_self_match},
            input_core_dims=[[self.dim]],
            vectorize=True, dask="parallelized", output_dtypes=[float]
        )

    def SampEn(self, da, m=2, r=None, tau=1):
        """Compute Sample Entropy over xarray objects."""
        return xr.apply_ufunc(
            self._sampen_1d, da,
            kwargs={"m": m, "r": r, "tau": tau},
            input_core_dims=[[self.dim]],
            vectorize=True, dask="parallelized", output_dtypes=[float]
        )

    # ----------------------------
    # Public interface for Cross-Entropy
    # ----------------------------
    def CrossApEn(self, da_x, da_y, m=2, r=None, tau=1, exclude_self_match=False):
        """Compute Cross-ApEn over xarray objects."""
        return xr.apply_ufunc(
            self._cross_apen_1d, da_x, da_y,
            kwargs={"m": m, "r": r, "tau": tau, "exclude_self_match": exclude_self_match},
            input_core_dims=[[self.dim],[self.dim]],
            vectorize=True, dask="parallelized", output_dtypes=[float]
        )

    def CrossSampEn(self, da_x, da_y, m=2, r=None, tau=1):
        """Compute Cross-SampEn over xarray objects."""
        return xr.apply_ufunc(
            self._cross_sampen_1d, da_x, da_y,
            kwargs={"m": m, "r": r, "tau": tau},
            input_core_dims=[[self.dim],[self.dim]],
            vectorize=True, dask="parallelized", output_dtypes=[float]
        )

    # ----------------------------
    # Multiscale entropy
    # ----------------------------
    def MSE(self, da, m=2, r=None, tau=1, max_scale=10, sample_entropy=True, exclude_self_match=False):
        """
        Compute Multiscale Entropy (MSE).
        
        Parameters
        ----------
        da : xarray.DataArray
            Input time series.
        m : int
            Embedding dimension.
        r : float or None
            Tolerance (if None, defaults to 0.2 * std at each scale).
        tau : int
            Time delay.
        max_scale : int
            Maximum scale factor.
        sample_entropy : bool
            If True, use Sample Entropy (SampEn); otherwise, use ApEn.
        exclude_self_match : bool
            If True, exclude self-matches (only for ApEn).
        
        Returns
        -------
        ndarray
            Entropy values at each scale.
        """
        x = np.asarray(da.values, float)
        results = []
        for scale in range(1, max_scale+1):
            y = self._coarse_grain(x, scale)
            if sample_entropy:
                val = self._sampen_1d(y, m, r, tau)
            else:
                val = self._apen_1d(y, m, r, tau, exclude_self_match)
            results.append(val)
        return np.array(results)
