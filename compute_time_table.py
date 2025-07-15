from pathlib import Path

import numpy as np

BASE_PATH = Path(__file__).parent / 'data'
BASE_PATH.mkdir(exist_ok=True)


times = [np.loadtxt(BASE_PATH / 'ex4_rsvd_p0q0_s25_alg_times.dat').T,
         np.loadtxt(BASE_PATH / 'ex4_rsvd_p1q0_s25_alg_times.dat').T]

with open(BASE_PATH / 'ex4_s25_times_and_solves.dat', 'w') as f:
    print('p r setup setupfine 0 0fine 1 1fine 2 2fine 3 3fine 4 4fine', file=f)
    for p in range(2):
        for r in range(len(times[0])):
            print(f'$p={p}$', f'$r={r}$', file=f, end='')
            for it in range(6):
                print('', f'{times[p][r, it]:.0f}', f'({1 + 2*(r+p) + it})', file=f, end='')
            print(file=f)
