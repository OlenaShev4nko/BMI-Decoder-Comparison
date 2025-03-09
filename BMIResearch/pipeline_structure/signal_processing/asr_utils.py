# This project includes code adapted from [DiGyt/asrpy](https://github.com/DiGyt/asrpy/blob/main/asrpy/asr_utils.py)
# Authors:  Nicolas Barascud
#           Dirk Gütlin <dirk.guetlin@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy.special import gamma, gammaincinv
from scipy.linalg import toeplitz
from scipy import signal
from scipy.spatial.distance import cdist, euclidean
from scipy import linalg
from numpy.linalg import pinv
from tqdm import tqdm


def block_covariance(data, window=128):
    """Compute blockwise covariance.

    Parameters
    ----------
    data : array, shape=(n_chans, n_samples)
        Input data (must be 2D)
    window : int
        Window size.

    Returns
    -------
    cov : array, shape=(n_blocks, n_chans, n_chans)
        Block covariance.
    """
    n_ch, n_times = data.shape
    U = np.zeros([len(np.arange(0, n_times - 1, window)), n_ch**2])
    data = data.T
    for k in range(0, window):
        idx_range = np.minimum(n_times - 1,
                               np.arange(k, n_times + k - 2, window))
        U = U + np.reshape(data[idx_range].reshape([-1, 1, n_ch]) *
                           data[idx_range].reshape(-1, n_ch, 1), U.shape)

    return np.array(U)


def numf(h, a, nb):
    """Get numerator B given impulse-response h of B/A and denominator A."""
    nh = np.max(h.size)
    xn = np.concatenate((1, np.zeros((1, nh - 1))), axis=None)
    impr = signal.lfilter(np.array([1.0]), a, xn)

    b = np.linalg.lstsq(
        toeplitz(impr, np.concatenate((1, np.zeros((1, nb))), axis=None)),
        h.T, rcond=None)[0].T

    return b

def denf(R, na):
    """Compute order NA denominator A from covariances R(0)...R(nr)."""
    nr = np.max(np.size(R))
    Rm = toeplitz(R[na:nr - 1], R[na:0:-1])
    Rhs = - R[na + 1:nr]
    A = np.concatenate(
        (1, np.linalg.lstsq(Rm, Rhs.T, rcond=None)[0].T), axis=None)
    return A


def ma_filter(N, X, Zi):
    """Run a moving average filter over the data.

    Parameters
    ----------
    N : int
        Length of the filter.
    X : array, shape=(n_channels, n_samples)
        The raw data.
    Zi : array
        The initial filter conditions.

    Returns
    -------
    X : array
        The filtered data.
    Zf : array
        The new fiter conditions.
    """
    if Zi is None:
        Zi = np.zeros([len(X), N])
    Y = np.concatenate([Zi, X], axis=1)
    M = Y.shape[-1]
    I_ = np.stack([np.arange(M - N),
                   np.arange(N, M)]).astype(int)
    S = (np.stack([-np.ones(M - N),
                   np.ones(M - N)]) / N)
    X = np.cumsum(np.multiply(Y[:, np.reshape(I_.T, -1)],
                              np.reshape(S.T, [-1])), axis=-1)
    X = X[:, 1::2]
    Zf = np.concatenate([-(X[:, -1] * N - Y[:, -N])[:, np.newaxis],
                         Y[:, -N + 1:]], axis=-1)
    return X, Zf


def geometric_median(X, tol=1e-5, max_iter=500):
    """Geometric median.

    This code is adapted from [2]_ using the Vardi and Zhang algorithm
    described in [1]_.

    Parameters
    ----------
    X : array, shape=(n_observations, n_variables)
        The data.
    tol : float
        Tolerance (default=1.e-5)
    max_iter : int
        Max number of iterations (default=500):

    Returns
    -------
    y1 : array, shape=(n_variables,)
        Geometric median over X.

    References
    ----------
    .. [1] Vardi, Y., & Zhang, C. H. (2000). The multivariate L1-median and
       associated data depth. Proceedings of the National Academy of Sciences,
       97(4), 1423-1426. https://doi.org/10.1073/pnas.97.4.1423
    .. [2] https://stackoverflow.com/questions/30299267/

    """
    y = np.mean(X, 0)  # initial value

    i = 0
    while i < max_iter:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1. / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < tol:
            return y1

        y = y1
        i += 1
    else:
        print(f"Geometric median could converge in {i} iterations "
              f"with a tolerance of {tol}")


def asr_process(data, sfreq, M, T, windowlen=0.5, lookahead=0.25, stepsize=32,
                maxdims=0.66, ab=None, R=None, Zi=None, cov=None, carry=None,
                return_states=False, method="euclid", mem_splits=3):
    """Apply the Artifact Subspace Reconstruction method to a data array.

    This function is used to clean multi-channel signal using the ASR method.
    The required inputs are the data matrix and the sampling rate of the data.

    `asr_process` can be used if you inted to apply ASR to a simple numpy
    array instead of a mne.io.Raw object. It is equivalent to the MATLAB
    implementation of `asr_process` (except for some small differences
    introduced by solvers for the eigenspace functions etc).

    Parameters
    ----------
    data : array, shape=(n_channels, n_samples)
        Raw data.
    sfreq : float
        The sampling rate of the data.
    M : array, shape=(n_channels, n_channels)
        The Mixing matrix (as fitted with asr_calibrate).
    T : array, shape=(n_channels, n_channels)
        The Threshold matrix (as fitted with asr_calibrate).
    windowlen : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but short enough to allow for several 1000 windows to compute
        statistics over (default=0.5).
    lookahead:
        Amount of look-ahead that the algorithm should use. Since the
        processing is causal, the output signal will be delayed by this
        amount. This value is in seconds and should be between 0 (no
        lookahead) and WindowLength/2 (optimal lookahead). The recommended
        value is WindowLength/2. Default: 0.25
    stepsize:
        The steps in which the algorithm will be updated. The larger this is,
        the faster the algorithm will be. The value must not be larger than
        WindowLength * SamplingRate. The minimum value is 1 (update for every
        sample) while a good value would be sfreq//3. Note that an update
        is always performed also on the first and last sample of the data
        chunk. Default: 32
    max_dims : float, int
        Maximum dimensionality of artifacts to remove. This parameter
        denotes the maximum number of dimensions which can be removed from
        each segment. If larger than 1, `int(max_dims)` will denote the
        maximum number of dimensions removed from the data. If smaller than 1,
        `max_dims` describes a fraction of total dimensions. Defaults to 0.66.
    ab : 2-tuple | None
        Coefficients (A, B) of an IIR filter that is used to shape the
        spectrum of the signal when calculating artifact statistics. The
        output signal does not go through this filter. This is an optional way
        to tune the sensitivity of the algorithm to each frequency component
        of the signal. The default filter is less sensitive at alpha and beta
        frequencies and more sensitive at delta (blinks) and gamma (muscle)
        frequencies. Defaults to None.
    R : array, shape=(n_channels, n_channels)
        Previous reconstruction matrix. Defaults to None.
    Zi : array
        Previous filter conditions. Defaults to None.
    cov : array, shape=([n_trials, ]n_channels, n_channels) | None
        Covariance. If None (default), then it is computed from ``X_filt``.
        If a 3D array is provided, the average covariance is computed from
        all the elements in it. Defaults to None.
    carry :
        Initial portion of the data that will be added to the current data.
        If None, data will be interpolated. Defaults to None.
    return_states : bool
        If True, returns a dict including the updated states {"M":M, "T":T,
        "R":R, "Zi":Zi, "cov":cov, "carry":carry}. Defaults to False.
    method : {'euclid', 'riemann'}
        Metric to compute the covariance matrix average. Currently, only
        euclidean ASR is supported.
    mem_splits : int
        Split the array in `mem_splits` segments to save memory.
    Returns
    -------
    clean : array, shape=(n_channels, n_samples)
        Clean data.
    state : dict
        Output ASR parameters {"M":M, "T":T, "R":R, "Zi":Zi, "cov":cov,
        "carry":carry}.

    """
    if method == "riemann":
        print("Riemannian ASR is not yet supported. Switching back to"
                      " Euclidean ASR.")
        method == "euclid"

    # calculate the the actual max dims based on the fraction parameter
    if maxdims < 1:
        maxdims = np.round(len(data) * maxdims)

    # set initial filter conditions of none was passed
    if Zi is None:
        _, Zi = yulewalk_filter(data, ab=ab, sfreq=sfreq,
                                zi=np.ones([len(data), 8]))

    # set the number of channels
    C, S = data.shape

    # set the number of windows
    N = np.round(windowlen * sfreq).astype(int)
    P = np.round(lookahead * sfreq).astype(int)

    # interpolate a portion of the data if no buffer was given
    if carry is None:
        carry = np.tile(2 * data[:, 0],
                        (P, 1)).T - data[:, np.mod(np.arange(P, 0, -1), S)]
    data = np.concatenate([carry, data], axis=-1)

    # splits = np.ceil(C*C*S*8*8 + C*C*8*s/stepsize + C*S*8*2 + S*8*5)...
    splits = mem_splits  # TODO: use this for parallelization MAKE IT A PARAM FIRST

    # loop over smaller segments of the data (for memory purposes)
    last_trivial = False
    last_R = None
    for i in range(splits):
        # set the current range
        i_range = np.arange(i * S // splits,
                            np.min([(i + 1) * S // splits, S]),
                            dtype=int)

        # filter the current window with yule-walker
        X, Zi = yulewalk_filter(data[:, i_range + P], sfreq=sfreq,
                                zi=Zi, ab=ab, axis=-1)
        # compute a moving average covariance
        Xcov, cov = \
            ma_filter(N,
                      np.reshape(np.multiply(np.reshape(X, (1, C, -1)),
                                             np.reshape(X, (C, 1, -1))),
                                 (C * C, -1)), cov)
        # set indices at which we update the signal
        update_at = np.arange(stepsize,
                              Xcov.shape[-1] + stepsize - 2,
                              stepsize)
        update_at = np.minimum(update_at, Xcov.shape[-1]) - 1
        # set the previous reconstruction matrix if none was assigned
        if last_R is None:
            update_at = np.concatenate([[0], update_at])
            last_R = np.eye(C)
        Xcov = np.reshape(Xcov[:, update_at], (C, C, -1))
        # loop through the updating intervals
        last_n = 0
        for j in range(len(update_at) - 1):
            # get the eigenvectors/values.For method 'riemann', this should
            # be replaced with PGA/ nonlinear eigenvalues
            D, V = np.linalg.eigh(Xcov[:, :, j])
            # determine which components to keep
            keep = np.logical_or(D < np.sum((T @ V) ** 2, axis=0),
                                 np.arange(C) + 1 < (C - maxdims))
            trivial = np.all(keep)
            # set the reconstruction matrix (ie. reconstructing artifact
            # components using the mixing matrix)
            if not trivial:
                inv = pinv(np.multiply(keep[:, np.newaxis], V.T @ M))
                R = np.real(M @ inv @ V.T)
            else:
                R = np.eye(C)

            # apply the reconstruction
            n = update_at[j] + 1
            if (not trivial) or (not last_trivial):
                subrange = i_range[np.arange(last_n, n)]

                # generate a cosine signal
                blend_x = np.pi * np.arange(1, n - last_n + 1) / (n - last_n)
                blend = (1 - np.cos(blend_x)) / 2

                # use cosine blending to replace data with reconstructed data
                tmp_data = data[:, subrange]
                data[:, subrange] = np.multiply(blend, R @ tmp_data) + \
                                    np.multiply(1 - blend, last_R @ tmp_data)  # noqa

            # set the parameters for the next iteration
            last_n, last_R, last_trivial = n, R, trivial
    # assign a new lookahead portion
    carry = np.concatenate([carry, data[:, -P:]])
    carry = carry[:, -P:]

    if return_states:
        return data[:, :-P], {"M": M, "T": T, "R": R, "Zi": Zi,
                              "cov": cov, "carry": carry}
    else:
        return data[:, :-P]


def polystab(a):
    """Polynomial stabilization.

    POLYSTAB(A), where A is a vector of polynomial coefficients,
    stabilizes the polynomial with respect to the unit circle;
    roots whose magnitudes are greater than one are reflected
    inside the unit circle.

    Parameters
    ----------
    a : array
        The vector of polynomial coefficients.

    Returns
    -------
    b : array
        The stabilized polynomial.

    Examples
    --------
    Convert a linear-phase filter into a minimum-phase filter with the same
    magnitude response.
    >>> h = fir1(25,0.4);               # Window-based FIR filter design
    >>> flag_linphase = islinphase(h)   # Determines if filter is linear phase
    >>> hmin = polystab(h) * norm(h)/norm(polystab(h));
    >>> flag_minphase = isminphase(hmin)# Determines if filter is min phase

    """
    v = np.roots(a)
    i = np.where(v != 0)
    vs = 0.5 * (np.sign(np.abs(v[i]) - 1) + 1)
    v[i] = (1 - vs) * v[i] + vs / np.conj(v[i])
    ind = np.where(a != 0)
    b = a[ind[0][0]] * np.poly(v)

    # Return only real coefficients if input was real:
    if not(np.sum(np.imag(a))):
        b = np.real(b)

    return b


def yulewalk(order, F, M):
    """Recursive filter design using a least-squares method.

    [B,A] = YULEWALK(N,F,M) finds the N-th order recursive filter
    coefficients B and A such that the filter:
    B(z)   b(1) + b(2)z^-1 + .... + b(n)z^-(n-1)
    ---- = -------------------------------------
    A(z)    1   + a(1)z^-1 + .... + a(n)z^-(n-1)
    matches the magnitude frequency response given by vectors F and M.
    The YULEWALK function performs a least squares fit in the time domain. The
    denominator coefficients {a(1),...,a(NA)} are computed by the so called
    "modified Yule Walker" equations, using NR correlation coefficients
    computed by inverse Fourier transformation of the specified frequency
    response H.
    The numerator is computed by a four step procedure. First, a numerator
    polynomial corresponding to an additive decomposition of the power
    frequency response is computed. Next, the complete frequency response
    corresponding to the numerator and denominator polynomials is evaluated.
    Then a spectral factorization technique is used to obtain the impulse
    response of the filter. Finally, the numerator polynomial is obtained by a
    least squares fit to this impulse response. For a more detailed
    explanation of the algorithm see [1]_.

    Parameters
    ----------
    order : int
        Filter order.
    F : array
        Normalised frequency breakpoints for the filter. The frequencies in F
        must be between 0.0 and 1.0, with 1.0 corresponding to half the sample
        rate. They must be in increasing order and start with 0.0 and end with
        1.0.
    M : array
        Magnitude breakpoints for the filter such that PLOT(F,M) would show a
        plot of the desired frequency response.

    References
    ----------
    .. [1] B. Friedlander and B. Porat, "The Modified Yule-Walker Method of
           ARMA Spectral Estimation," IEEE Transactions on Aerospace
           Electronic Systems, Vol. AES-20, No. 2, pp. 158-173, March 1984.

    Examples
    --------
    Design an 8th-order lowpass filter and overplot the desired
    frequency response with the actual frequency response:
    >>> f = [0, .6, .6, 1]         # Frequency breakpoints
    >>> m = [1, 1, 0, 0]           # Magnitude breakpoints
    >>> [b, a] = yulewalk(8, f, m) # Filter design using least-squares method

    """
    F = np.asarray(F)
    M = np.asarray(M)
    npt = 512
    lap = np.fix(npt / 25).astype(int)
    mf = F.size
    npt = npt + 1  # For [dc 1 2 ... nyquist].
    Ht = np.array(np.zeros((1, npt)))
    nint = mf - 1
    df = np.diff(F)

    nb = 0
    Ht[0][0] = M[0]
    for i in range(nint):
        if df[i] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.fix(F[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)
        if ne == nb:
            inc = 0
        else:
            inc = (j - nb) / (ne - nb)

        Ht[0][nb:ne + 1] = np.array(inc * M[i + 1] + (1 - inc) * M[i])
        nb = ne + 1

    Ht = np.concatenate((Ht, Ht[0][-2:0:-1]), axis=None)
    n = Ht.size
    n2 = np.fix((n + 1) / 2)
    nb = order
    nr = 4 * order
    nt = np.arange(0, nr)

    # compute correlation function of magnitude squared response
    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[0:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr - 1)))   # pick NR correlations  # noqa

    # Form window to be used in extracting the right "wing" of two-sided
    # covariance sequence
    Rwindow = np.concatenate(
        (1 / 2, np.ones((1, int(n2 - 1))), np.zeros((1, int(n - n2)))),
        axis=None)
    A = polystab(denf(R, order))  # compute denominator

    # compute additive decomposition
    Qh = numf(np.concatenate((R[0] / 2, R[1:nr]), axis=None), A, order)

    # compute impulse response
    _, Ss = 2 * np.real(signal.freqz(Qh, A, worN=n, whole=True))

    hh = np.fft.ifft(
        np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss, dtype=np.complex128))))  # noqa
    )
    B = np.real(numf(hh[0:nr], A, nb))

    return B, A


def yulewalk_filter(X, sfreq, zi=None, ab=None, axis=-1):
    """Yulewalk filter.

    Parameters
    ----------
    X : array, shape = (n_channels, n_samples)
        Data to filter.
    sfreq : float
        Sampling frequency.
    zi : array, shape=(n_channels, filter_order)
        Initial conditions.
    a, b : 2-tuple | None
        Coefficients of an IIR filter that is used to shape the spectrum of
        the signal when calculating artifact statistics. The output signal
        does not go through this filter. This is an optional way to tune the
        sensitivity of the algorithm to each frequency component of the
        signal. The default filter is less sensitive at alpha and beta
        frequencies and more sensitive at delta (blinks) and gamma (muscle)
        frequencies.
    axis : int
        Axis to filter on (default=-1, corresponding to samples).

    Returns
    -------
    out : array
        Filtered data.
    zf :  array, shape=(n_channels, filter_order)
        Output filter state.
    """
    # Set default IIR filter coefficients
    if ab is None:
        F = np.array([0, 2, 3, 13, 16, 40, np.minimum(
            80.0, (sfreq / 2.0) - 1.0), sfreq / 2.0]) * 2.0 / sfreq
        M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
        B, A = yulewalk(8, F, M)
    else:
        A, B = ab

    # apply the signal shaping filter and initialize the IIR filter state
    if zi is None:
        out = signal.lfilter(B, A, X, axis=axis)
        zf = None
    else:
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)

    return out, zf


def fit_eeg_distribution(X, min_clean_fraction=0.25, max_dropout_fraction=0.1,
                         fit_quantiles=[0.022, 0.6], step_sizes=[0.01, 0.01],
                         shape_range=np.arange(1.7, 3.5, 0.15)):
    """Estimate the mean and SD of clean EEG from contaminated data.

    This function estimates the mean and standard deviation of clean EEG from
    a sample of amplitude values (that have preferably been computed over
    short windows) that may include a large fraction of contaminated samples.
    The clean EEG is assumed to represent a generalized Gaussian component in
    a mixture with near-arbitrary artifact components. By default, at least
    25% (`min_clean_fraction`) of the data must be clean EEG, and the rest
    can be contaminated. No more than 10% (`max_dropout_fraction`) of the
    data is allowed to come from contaminations that cause lower-than-EEG
    amplitudes (e.g., sensor unplugged). There are no restrictions on
    artifacts causing larger-than-EEG amplitudes, i.e., virtually anything is
    handled (with the exception of a very unlikely type of distribution that
    combines with the clean EEG samples into a larger symmetric generalized
    Gaussian peak and thereby "fools" the estimator). The default parameters
    should work for a wide range of applications but may be adapted to
    accommodate special circumstances.
    The method works by fitting a truncated generalized Gaussian whose
    parameters are constrained by `min_clean_fraction`,
    `max_dropout_fraction`, `fit_quantiles`, and `shape_range`. The fit is
    performed by a grid search that always finds a close-to-optimal solution
    if the above assumptions are fulfilled.

    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        EEG data, possibly containing artifacts.
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.25).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG
        (default=0.1).
    fit_quantiles : 2-tuple
        Quantile range [lower,upper] of the truncated generalized Gaussian
        distribution that shall be fit to the EEG contents (default=[0.022
        0.6]).
    step_sizes : 2-tuple
        Step size of the grid search; the first value is the stepping of the
        lower bound (which essentially steps over any dropout samples), and
        the second value is the stepping over possible scales (i.e., clean-
        data quantiles) (default=[0.01, 0.01]).
    beta : array
        Range that the clean EEG distribution's shape parameter beta may take.

    Returns
    -------
    mu : array
        Estimated mean of the clean EEG distribution.
    sig : array
        Estimated standard deviation of the clean EEG distribution.
    alpha : float
        Estimated scale parameter of the generalized Gaussian clean EEG
        distribution.
    beta : float
        Estimated shape parameter of the generalized Gaussian clean EEG
        distribution.

    """
    # sort data so we can access quantiles directly
    X = np.sort(X)
    n = len(X)

    # compute z bounds for the truncated standard generalized Gaussian pdf and
    # pdf rescaler
    quants = np.array(fit_quantiles)
    zbounds = []
    rescale = []
    for b in range(len(shape_range)):
        gam = gammaincinv(
            1 / shape_range[b], np.sign(quants - 1 / 2) * (2 * quants - 1))
        zbounds.append(np.sign(quants - 1 / 2) * gam ** (1 / shape_range[b]))
        rescale.append(shape_range[b] / (2 * gamma(1 / shape_range[b])))

    # determine the quantile-dependent limits for the grid search
    # we can generally skip the tail below the lower quantile
    lower_min = np.min(quants)
    # maximum width is the fit interval if all data is cleanT
    max_width = np.diff(quants)
    # minimum width of the fit interval, as fraction of data
    min_width = min_clean_fraction * max_width

    # Build quantile interval matrix
    cols = np.arange(lower_min,
                     lower_min + max_dropout_fraction + step_sizes[0] * 1e-9,
                     step_sizes[0])
    cols = np.round(n * cols).astype(int)
    rows = np.arange(0, int(np.round(n * max_width)))
    newX = np.zeros((len(rows), len(cols)))
    for i, c in enumerate(range(len(rows))):
        newX[i] = X[c + cols]

    # subtract baseline value for each interval
    X1 = newX[0, :]
    newX = newX - X1

    opt_val = np.inf
    opt_lu = np.inf
    opt_bounds = np.inf
    opt_beta = np.inf
    gridsearch = np.round(n * np.arange(max_width, min_width, -step_sizes[1]))
    for m in gridsearch.astype(int):
        mcurr = m - 1
        nbins = int(np.round(3 * np.log2(1 + m / 2)))
        cols = nbins / newX[mcurr]
        H = newX[:m] * cols

        hist_all = []
        for ih in range(len(cols)):
            histcurr = np.histogram(H[:, ih], bins=np.arange(0, nbins + 1))
            hist_all.append(histcurr[0])
        hist_all = np.array(hist_all, dtype=int).T
        hist_all = np.vstack((hist_all, np.zeros(len(cols), dtype=int)))
        logq = np.log(hist_all + 0.01)

        # for each shape value...
        for k, b in enumerate(shape_range):
            bounds = zbounds[k]
            x = bounds[0] + np.arange(0.5, nbins + 0.5) / nbins * np.diff(bounds)  # noqa:E501
            p = np.exp(-np.abs(x) ** b) * rescale[k]
            p = p / np.sum(p)

            # calc KL divergences
            kl = np.sum(p * (np.log(p) - logq[:-1, :].T), axis=1) + np.log(m)

            # update optimal parameters
            min_val = np.min(kl)
            idx = np.argmin(kl)
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = shape_range[k]
                opt_bounds = bounds
                opt_lu = [X1[idx], X1[idx] + newX[m - 1, idx]]

    # recover distribution parameters at optimum
    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta

    # calculate the distribution's standard deviation from alpha and beta
    sig = np.sqrt((alpha ** 2) * gamma(3 / beta) / gamma(1 / beta))

    return mu, sig, alpha, beta


def clean_windows(X, sfreq, max_bad_chans=0.2, zthresholds=[-3.5, 5],
                  win_len=.5, win_overlap=0.66, min_clean_fraction=0.25,
                  max_dropout_fraction=0.1):
    """Remove periods with abnormally high-power content from continuous data.

    This function cuts segments from the data which contain high-power
    artifacts. Specifically, only windows are retained which have less than a
    certain fraction of "bad" channels, where a channel is bad in a window if
    its power is above or below a given upper/lower threshold (in standard
    deviations from a robust estimate of the EEG power distribution in the
    channel).

    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        Continuous data set, assumed to be appropriately high-passed (e.g. >
        1Hz or 0.5Hz - 2.0Hz transition band)
    max_bad_chans : float
        The maximum number or fraction of bad channels that a retained window
        may still contain (more than this and it is removed). Reasonable range
        is 0.05 (very clean output) to 0.3 (very lax cleaning of only coarse
        artifacts) (default=0.2).
    zthresholds : 2-tuple
        The minimum and maximum standard deviations within which the power of
        a channel must lie (relative to a robust estimate of the clean EEG
        power distribution in the channel) for it to be considered "not bad".
        (default=[-3.5, 5]).

    The following are detail parameters that usually do not have to be tuned.
    If you can't get the function to do what you want, you might consider
    adapting these to your data.

    win_len : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but not shorter than half a cycle of the high-pass filter that was
        used. Default: 1.
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are
        going to be missed, but is slower (default=0.66).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG.
        (default=0.25)
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.1).

    Returns
    -------
    clean : array, shape=(n_channels, n_samples)
        Dataset with bad time periods removed.
    sample_mask : boolean array, shape=(1, n_samples)
        Mask of retained samples (logical array).

    """
    assert 0 < max_bad_chans < 1, "max_bad_chans must be a fraction !"

    # set internal variables
    truncate_quant = [0.0220, 0.6000]
    step_sizes = [0.01, 0.01]
    shape_range = np.arange(1.7, 3.5, 0.15)
    max_bad_chans = np.round(X.shape[0] * max_bad_chans)

    # set data indices
    [nc, ns] = X.shape
    N = int(win_len * sfreq)
    offsets = np.int_(np.round(np.arange(0, ns - N, (N * (1 - win_overlap)))))
    print('[ASR] Determining channel-wise rejection thresholds')

    wz = np.zeros((nc, len(offsets)))
    for ichan in range(nc):

        # compute root mean squared amplitude
        x = X[ichan, :] ** 2
        Y = np.array([np.sqrt(np.sum(x[o:o + N]) / N) for o in offsets])

        # fit a distribution to the clean EEG part
        mu, sig, alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction, truncate_quant,
            step_sizes, shape_range)
        # calculate z scores
        wz[ichan] = (Y - mu) / sig

    # sort z scores into quantiles
    wz[np.isnan(wz)] = np.inf  # Nan to inf
    swz = np.sort(wz, axis=0)

    # determine which windows to remove
    if np.max(zthresholds) > 0:
        mask1 = swz[-(np.int64(max_bad_chans) + 1), :] > np.max(zthresholds)
    if np.min(zthresholds) < 0:
        mask2 = (swz[1 + np.int64(max_bad_chans - 1), :] < np.min(zthresholds))

    # combine the two thresholds
    remove_mask = np.logical_or.reduce((mask1, mask2))
    removed_wins = np.where(remove_mask)

    # reconstruct the samples to remove
    sample_maskidx = []
    for i in range(len(removed_wins[0])):
        if i == 0:
            sample_maskidx = np.arange(
                offsets[removed_wins[0][i]], offsets[removed_wins[0][i]] + N)
        else:
            sample_maskidx = np.vstack((
                sample_maskidx,
                np.arange(offsets[removed_wins[0][i]],
                          offsets[removed_wins[0][i]] + N)
            ))

    # delete the bad chunks from the data
    sample_mask2remove = np.unique(sample_maskidx)
    if sample_mask2remove.size:
        clean = np.delete(X, sample_mask2remove, 1)
        sample_mask = np.ones((1, ns), dtype=bool)
        sample_mask[0, sample_mask2remove] = False
    else:
        sample_mask = np.ones((1, ns), dtype=bool)

    return clean, sample_mask


def asr_calibrate(X, sfreq, cutoff=20, blocksize=100, win_len=0.5,
                  win_overlap=0.66, max_dropout_fraction=0.1,
                  min_clean_fraction=0.25, ab=None, method='euclid'):
    """Calibration function for the Artifact Subspace Reconstruction method.

    This function can be used if you inted to apply ASR to a simple numpy
    array instead of a mne.io.Raw object. It is equivalent to the MATLAB
    implementation of asr_calibrate (except for some small differences
    introduced by solvers for the eigenspace functions etc).

    The input to this data is a multi-channel time series of calibration data.
    In typical uses the calibration data is clean resting EEG data of ca. 1
    minute duration (can also be longer). One can also use on-task data if the
    fraction of artifact content is below the breakdown point of the robust
    statistics used for estimation (50% theoretical, ~30% practical). If the
    data has a proportion of more than 30-50% artifacts then bad time windows
    should be removed beforehand. This data is used to estimate the thresholds
    that are used by the ASR processing function to identify and remove
    artifact components.

    The calibration data must have been recorded for the same cap design from
    which data for cleanup will be recorded, and ideally should be from the
    same session and same subject, but it is possible to reuse the calibration
    data from a previous session and montage to the extent that the cap is
    placed in the same location (where loss in accuracy is more or less
    proportional to the mismatch in cap placement).

    The calibration data should have been high-pass filtered (for example at
    0.5Hz or 1Hz using a Butterworth IIR filter).

    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        *zero-mean* (e.g., high-pass filtered) and reasonably clean EEG of not
        much less than 30 seconds (this method is typically used with 1 minute
        or more).
    sfreq : float
        Sampling rate of the data, in Hz.
    cutoff: float
        Standard deviation cutoff for rejection. X portions whose variance
        is larger than this threshold relative to the calibration data are
        considered missing data and will be removed. Defaults to 20
        (In EEGLab's `clean_rawdata` the original threshold was set to 5, but
        it is widely recommended to use a value higher than 20).
    blocksize : int
        Block size for calculating the robust data covariance and thresholds,
        in samples; allows to reduce the memory and time requirements of the
        robust estimators by this factor (down to n_chans x n_chans x
        n_samples x 16 / blocksize bytes) (default=100).
    win_len : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but short enough to allow for several 1000 windows to compute
        statistics over (default=0.5).
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are
        going to be missed, but is slower (default=0.66).
    max_dropout_fraction : float
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation (default=0.1).
    min_clean_fraction : float
        Minimum fraction of windows that need to be clean, used for threshold
        estimation (default=0.25).
    ab : 2-tuple | None
        Coefficients (A, B) of an IIR filter that is used to shape the
        spectrum of the signal when calculating artifact statistics. The
        output signal does not go through this filter. This is an optional way
        to tune the sensitivity of the algorithm to each frequency component
        of the signal. The default filter is less sensitive at alpha and beta
        frequencies and more sensitive at delta (blinks) and gamma (muscle)
        frequencies. Defaults to None.
    method : {'euclid', 'riemann'}
        Metric to compute the covariance matrix average. For now, only
        euclidean ASR is supported.

    Returns
    -------
    M : array
        Mixing matrix.
    T : array
        Threshold matrix.

    """
    if method == "riemann":
        print("Riemannian ASR is not yet supported. Switching back to"
                      " Euclidean ASR.")
        method == "euclid"

    print('[ASR] Calibrating...')

    # set number of channels and number of samples
    [nc, ns] = X.shape

    # filter the data
    X, _zf = yulewalk_filter(X, sfreq, ab=ab)

    # window length for calculating thresholds
    N = int(np.round(win_len * sfreq))

    # get block covariances
    U = block_covariance(X, window=blocksize)

    # get geometric median for each block
    # Note: riemann mode is not yet supported, else this could be:
    # Uavg = pyriemann.utils.mean_covariance(U, metric='riemann')
    Uavg = geometric_median(U.reshape((-1, nc * nc)) / blocksize)
    Uavg = Uavg.reshape((nc, nc))

    # get the mixing matrix M
    M = linalg.sqrtm(np.real(Uavg))

    # sort the get the sorted eigenvecotors/eigenvalues
    # riemann is not yet supported, else this could be PGA/nonlinear eigenvs
    D, Vtmp = linalg.eigh(M)
    V = Vtmp[:, np.argsort(D)]  # I think numpy sorts them automatically

    # get the threshold matrix T
    x = np.abs(np.dot(V.T, X))
    offsets = np.int_(np.arange(0, ns - N, np.round(N * (1 - win_overlap))))

    # go through all the channels and fit the EEG distribution
    mu = np.zeros(nc)
    sig = np.zeros(nc)
    for ichan in reversed(range(nc)):
        rms = x[ichan, :] ** 2
        Y = []
        for o in offsets:
            Y.append(np.sqrt(np.sum(rms[o:o + N]) / N))
        mu[ichan], sig[ichan], alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction)
    T = np.dot(np.diag(mu + cutoff * sig), V.T)

    print('[ASR] Calibration done.')
    return M, T
    
    
def apply_asr(data_chanks_list_train, sfreq, M, T):
    print('[apply_asr]')
    print('data_chanks_list_train shape = ', data_chanks_list_train.shape)
    final_train_set = []
    for chank_df in tqdm(data_chanks_list_train):
        asr_data = asr_process(chank_df, sfreq, M, T)
        final_train_set.append(asr_data)
    return np.array(final_train_set)



# if __name__ == '__main__':
