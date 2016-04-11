def autocorrelation(spike_times, bin_width=5e-3, width=1e-1, T=None):
    """Given the sorted spike train 'spike_times' return the
    autocorrelation histogram, as well as the bin edges (including the
    rightmost one). The bin size is specified by 'bin_width', and lags are
    required to fall within the interval [-width, width]. The algorithm is
    partly inspired on the Brian function with the same name."""

    d = []                    # Distance between any two spike times
    n_sp = alen(spike_times)  # Number of spikes in the input spike train

    i, j = 0, 0
    for t in spike_times:
        # For each spike we only consider those spikes times that are at most
        # at a 'width' time lag. This requires finding the indices
        # associated with the limiting spikes.
        while i < n_sp and spike_times[i] < t - width:
            i += 1
        while j < n_sp and spike_times[j] < t + width:
            j += 1
        # Once the relevant spikes are found, add the time differences
        # to the list
        d.extend(spike_times[i:j] - t)


    n_b = int( ceil(width / bin_width) )  # Num. edges per side
    # Define the edges of the bins (including rightmost bin)
    b = linspace(-width, width, 2 * n_b, endpoint=True)
    h = histogram(d, bins=b, new=True)
    H = h[0] # number of entries per bin

    # Compute the total duration, if it was not given
    # (spike trains are assumed to be sorted sequences)
    if T is None:
        T = spike_times[-1] - spike_times[0] # True for T >> 1/r

    # The sample space gets smaller as spikes are closer to the boundaries.
    # We have to take into account this effect.
    W = T - bin_width * abs( arange(n_b - 1, -n_b, -1) )

    return ( H/W - n_sp**2 * bin_width / (T**2), b)

