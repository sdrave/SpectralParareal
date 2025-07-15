from pathlib import Path

import numpy as np

from experiments import problem1, problem2, problem3

BASE_PATH = Path(__file__).parent / 'data'
BASE_PATH.mkdir(exist_ok=True)


def write_values(m, X, values, f):
    for i_t, row in list(enumerate(values))[::2 * m.time_stepper.nt//100]:
        for x, v in list(zip(X, row))[::2]:
            print(f'{x:.5}', f'{i_t * m.T / m.time_stepper.nt:.5f}', f'{v:.5f}', file=f)
        print(file=f)



ex1_path = BASE_PATH / 'ex1_solution'
ex1_path.mkdir(exist_ok=True)

for neumann in [False, True]:
    for k in range(5):
        m = problem1(2**k, neumann)
        U = m.solve()

        X = np.linspace(0, 1, 101)
        if neumann:
            values = U.to_numpy()
        else:
            values = np.zeros((len(U), 101))
            values[:,1:-1] = U.to_numpy()

        with open(ex1_path / f'{"neu" if neumann else "dir"}_k{k}.dat', 'w') as f:
            write_values(m, X, values, f)


ex2_path = BASE_PATH / 'ex2_solution'
ex2_path.mkdir(exist_ok=True)
m = problem2(1, neumann=True)
U = m.solve()

X = np.linspace(0, 1, 101)
values = U.to_numpy()

with open(ex2_path / 'solution.dat', 'w') as f:
    write_values(m, X, values, f)


ex3_path = BASE_PATH / 'ex3_solution'
ex3_path.mkdir(exist_ok=True)
m = problem3(1, False)
U = m.solve()
U = m.visualizer.add_dirichlet_dofs(U)

grid = m.visualizer.orig.grid
centers = grid.centers(2)
subentities = grid.subentities(0, 2)
np.savetxt(ex3_path / 'subentities.dat', subentities, fmt='%.5e')
for t in (0, 0.25, 0.5, 0.75, 1.):
    idx = int((len(U)-1)*t)
    points = np.hstack((centers, U[idx].to_numpy().T))
    np.savetxt(ex3_path / (f'points{t:.2f}'.replace('.', '_') + '.dat'), points, fmt='%.5e')
