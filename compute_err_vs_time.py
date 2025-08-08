from pathlib import Path

import numpy as np

BASE_PATH = Path(__file__).parent / 'data'
BASE_PATH.mkdir(exist_ok=True)


for err_path in BASE_PATH.glob('*_max_errs.dat'):
    base_name = err_path.name.removesuffix('_max_errs.dat')
    time_path = BASE_PATH / f'{base_name}_alg_times.dat'

    print(f'Processing {base_name}')
    errs = np.loadtxt(err_path, ndmin=2)
    times = np.loadtxt(time_path, ndmin=2)
    assert times.shape[0] == errs.shape[0] + 1
    assert times.shape[1] == errs.shape[1]

    with open(BASE_PATH / f'{base_name}_errs_vs_time.dat', 'w') as f:
        for j in range(errs.shape[1]):
            for i in range(errs.shape[0]):
                err = errs[i, j]
                time = times[i+1, j]
                print(f'{time:.4e}', f'{err:.4e}', chr(ord('a') + j), file=f)
