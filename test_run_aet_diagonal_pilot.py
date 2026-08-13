from types import SimpleNamespace

import numpy as np

from run_aet_diagonal_pilot import AET_CHANNELS, recovery_metrics


def test_aet_whitening_mask_is_diagnostic_only():
    shape = (len(AET_CHANNELS), 1, 3)
    noise = np.ones(shape)
    galactic = np.ones(shape)
    truth_total = noise + galactic
    observed = truth_total.copy()
    observed[:, :, 1] = 200.0
    counts = np.ones(shape)
    whitening_mask = np.ones(shape, dtype=bool)
    whitening_mask[:, :, 1] = False
    posterior = SimpleNamespace(
        noise_median=noise,
        galactic_median=galactic,
        total_median=truth_total,
        total_lower=0.9 * truth_total,
        total_upper=1.1 * truth_total,
    )
    metrics = recovery_metrics(
        observed,
        counts,
        noise,
        galactic,
        posterior,
        whitening_mask=whitening_mask,
    )
    for channel in ("a", "e", "t"):
        assert metrics[f"{channel}_fit_effective_cells"] == 3.0
        assert metrics[f"{channel}_whitening_effective_cells"] == 2.0
        assert metrics[f"{channel}_continuum_mean_z2"] == 1.0
