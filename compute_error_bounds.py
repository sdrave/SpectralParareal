import re
from pathlib import Path

import numpy as np

BASE_PATH = Path(__file__).parent / 'data'

for err_path in BASE_PATH.glob('*rsvd_*_max_errs.dat'):
    if 'p0' in err_path.name:
        continue
    ex_base_name = err_path.name.replace('_max_errs.dat', '')
    svals_file = re.sub('rsvd_p.q.', 'svals', ex_base_name) + '.dat'
    max_updates = np.loadtxt(BASE_PATH / (ex_base_name + '_max_updates.dat')).T  # (r, k)
    R, K = max_updates.shape

    svals = np.loadtxt(BASE_PATH / svals_file)  # (r, n)
    N = svals.shape[1]

    updates = [np.loadtxt(BASE_PATH / f'{ex_base_name}_updates_k{k}.dat') for k in range(R)]
    updates = np.array(updates)  # (r, k, n)
    assert np.all(np.max(updates, axis=2) == max_updates)

    rsvd_svals = [np.loadtxt(BASE_PATH / f'{ex_base_name}_svals_k{k}.dat', ndmin=2) for k in range(R)]  # (max r, r, n)

    max_errs = np.loadtxt(BASE_PATH / (ex_base_name + '_max_errs.dat')).T  # (r, k)

    max_bounds = np.zeros((R, K))
    max_bounds_rsvd = np.zeros((R, K))
    for r in range(R):
        delta = np.max(svals[0, :])
        delta_rsvd = np.max(rsvd_svals[r][0, :])
        eps = np.max(svals[r, :])
        eps_rsvd = np.max(rsvd_svals[r][r, :])
        for k in range(K):
            max_bounds[r, k] = eps / (1 - delta) * max_updates[r, k]
            max_bounds_rsvd[r, k] = eps_rsvd / (1 - delta_rsvd) * max_updates[r, k]
    max_effs = max_errs / max_bounds
    max_effs_rsvd = max_errs / max_bounds_rsvd


    bounds = np.zeros((R, K, N))
    bounds_rsvd = np.zeros((R, K, N))
    for r in range(R):
        delta = np.max(svals[0, :])
        delta_rsvd = np.max(rsvd_svals[r][0, :]) if rsvd_svals[r].shape[0] > 0 else np.nan
        eps = np.max(svals[r, :])
        eps_rsvd = np.max(rsvd_svals[r][r, :]) if rsvd_svals[r].shape[0] > r else np.nan
        for k in range(K):
            for n in range(N):
                for m in range(1, n):
                    bounds[r, k, n] += eps * delta**(n-m-1) * updates[r, k, m]
                    bounds_rsvd[r, k, n] += eps_rsvd * delta_rsvd**(n-m-1) * updates[r, k, m]
    bounds = np.max(bounds, axis=2)
    bounds_rsvd = np.max(bounds_rsvd, axis=2)
    effs = max_errs / bounds
    effs_rsvd = max_errs / bounds_rsvd

    np.savetxt(BASE_PATH / (ex_base_name + '_max_bounds.dat'), max_bounds.T, '%.4e',
               header='columns=rank, rows=iteration')
    np.savetxt(BASE_PATH / (ex_base_name + '_max_bounds_rsvd.dat'), max_bounds_rsvd.T, '%.4e',
               header='columns=rank, rows=iteration')
    np.savetxt(BASE_PATH / (ex_base_name + '_bounds.dat'), bounds.T, '%.4e',
               header='columns=rank, rows=iteration')
    np.savetxt(BASE_PATH / (ex_base_name + '_bounds_rsvd.dat'), bounds_rsvd.T, '%.4e',
               header='columns=rank, rows=iteration')

    np.savetxt(BASE_PATH / (ex_base_name + '_max_effs.dat'), max_effs.T, '%.4e',
               header='columns=rank, rows=iteration')
    np.savetxt(BASE_PATH / (ex_base_name + '_max_effs_rsvd.dat'), max_effs_rsvd.T, '%.4e',
               header='columns=rank, rows=iteration')
    np.savetxt(BASE_PATH / (ex_base_name + '_effs.dat'), effs.T, '%.4e',
               header='columns=rank, rows=iteration')
    np.savetxt(BASE_PATH / (ex_base_name + '_effs_rsvd.dat'), effs_rsvd.T, '%.4e',
               header='columns=rank, rows=iteration')
