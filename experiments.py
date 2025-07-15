import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from heatsink import heatsink_model
from helpers import remove_dirichlet
from mpi4py import MPI
from pararb import ParaRB, PararealAPrioriCoarse, PararealEuler, PararealNoCoarse, TimeSlice, exact_svd
from pymor.basic import (
    ConstantFunction,
    ExpressionFunction,
    InstationaryProblem,
    LineDomain,
    RectDomain,
    StationaryProblem,
    discretize_instationary_cg,
    gram_schmidt,
    set_log_levels,
)
from tqdm import tqdm as real_tqdm

BASE_PATH = Path(__file__).parent / 'data'
BASE_PATH.mkdir(exist_ok=True)

set_log_levels({'pymor': 'WARN'})

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def solve(m):
    tic = perf_counter()
    ts = TimeSlice.from_model(m)
    if rank == 0:
        initial_data = m.initial_data.as_range_array()
    else:
        initial_data = np.empty(m.solution_space.dim)
        comm.Recv(initial_data, source=rank-1)
        initial_data = m.solution_space.from_numpy(initial_data)

    U = ts.solve(initial_data)

    if rank+1 < comm.size:
        comm.Send(U[-1].to_numpy(), dest=rank+1)

    return U, perf_counter() - tic


def run_experiment(setup, K, ex_id, same_m=True):
    max_errs, max_updates, solve_times, alg_times = [], [], [], []
    solution = None
    for k in tqdm(range(K), ex_id):
        m, alg = setup(k)
        comm.barrier()
        if solution is None or not same_m:
            solution, local_t_solve = solve(m)
            local_t_solve = np.array([local_t_solve])

        comm.barrier()
        Us, Ucont, local_t_alg = alg.run()

        local_errors = np.array([(solution - u).norm() for u in Ucont]) + 1e-15 # avoid trouble in log plots
        local_updates = np.array([Us[0].norm().item()]
                                 + [(Us[i+1] - Us[i]).norm().item() for i in range(len(Us)-1)])

        errors, updates, t_solve, t_alg = None, None, None, None
        if rank == 0:
            errors = np.empty((comm.size,) + local_errors.shape)
            updates = np.empty((comm.size,) + local_updates.shape)
            t_solve = np.empty(comm.size)
            t_alg = np.empty((comm.size,) + local_t_alg.shape)
            assert errors.ndim == 3
            assert t_alg.ndim == 2
        comm.Gather(local_errors,   errors,  root=0)
        comm.Gather(local_updates,  updates,  root=0)
        comm.Gather(local_t_solve,  t_solve, root=0)
        comm.Gather(local_t_alg,    t_alg,   root=0)

        if rank == 0:
            errors = np.hstack(errors)
            assert errors.ndim == 2
            errors = np.hstack([np.full((len(errors), 1), 1e-15), errors])  # add error at t=0
            np.savetxt(BASE_PATH / f'ex{ex_id}_s{comm.size}_errs_k{k}.dat',
                       np.hstack([np.linspace(0, m.T, len(errors[0])).reshape((-1,1)), errors.T]), fmt='%.4e',
                       header='columns=iterations, rows=time')

            np.savetxt(BASE_PATH / f'ex{ex_id}_s{comm.size}_updates_k{k}.dat',
                       updates.T, fmt='%.4e',
                       header='columns=time, rows=iteration')

            max_errs.append(np.max(errors, axis=1))
            max_updates.append(np.max(updates, axis=0))
            solve_times.append(np.max(t_solve))
            alg_times.append(np.max(t_alg, axis=0))

        if hasattr(alg, 'svals'):
            local_svals = alg.svals
            assert local_svals.ndim == 1
            svals = None
            if rank == 0:
                svals = np.empty((comm.size, len(local_svals)))
            comm.Gather(local_svals, svals, root=0)

            if rank == 0:
                svals = svals.T
                np.savetxt(BASE_PATH / f'ex{ex_id}_s{comm.size}_svals_k{k}.dat', svals,
                           fmt='%.4e', header='columns=time slice, rows=singular values')

    if rank == 0:
        max_errs = np.array(max_errs)
        np.savetxt(BASE_PATH / f'ex{ex_id}_s{comm.size}_max_errs.dat', max_errs.T, fmt='%.4e',
                   header='columns=k, rows=iterations')

        max_updates = np.array(max_updates)
        np.savetxt(BASE_PATH / f'ex{ex_id}_s{comm.size}_max_updates.dat', max_updates.T, fmt='%.4e',
                   header='columns=k, rows=iterations')

        solve_times = np.array(solve_times)
        np.savetxt(BASE_PATH / f'ex{ex_id}_s{comm.size}_solve_times.dat', solve_times, fmt='%.4e')

        alg_times = np.array(alg_times)
        np.savetxt(BASE_PATH / f'ex{ex_id}_s{comm.size}_alg_times.dat', alg_times.T, fmt='%.4e',
                   header='columns=k, rows=iterations')


def run_euler_experiment(ex_id, problem_factory):
    m = problem_factory()
    ex_id = f'{ex_id}_euler'
    def setup(k):
        return m, PararealEuler(m)
    run_experiment(setup, 1, ex_id)


def run_pararb_experiment(ex_id, problem_factory, K, svd_method='rand', p=None, q=None):
    m = problem_factory()
    def setup(k):
        return m, ParaRB(m, k, svd_method=svd_method, p=p, q=q, homogeneous=False)
    ex_id = f'{ex_id}_rsvd_p{p}q{q}' if svd_method == 'rand' else f'{ex_id}_svd'
    run_experiment(setup, K, ex_id)


def run_svals_experiment(ex_id, problem_factory, num_svals):
    m = problem_factory()
    ex_id = f'{ex_id}_svals'
    ts = TimeSlice.from_model(m)
    for _ in tqdm(range(1), ex_id):
        U, s, V = exact_svd(ts.transfer_operator, num_svals=num_svals)

    local_svals = s
    svals = None
    if rank == 0:
        svals = np.empty((comm.size, num_svals))
    comm.Gather(local_svals, svals, root=0)

    if rank == 0:
        svals = svals.T
        np.savetxt(BASE_PATH / f'ex{ex_id}_s{comm.size}.dat', svals,
                   fmt='%.4e', header='columns=time slice, rows=singular values')


def run_solve_experiment(ex_id, problem_factory):
    if rank != 0:
        return

    m = problem_factory()
    ex_id = f'{ex_id}_rank0_solve'

    for _ in tqdm(range(1), ex_id):
        tic = perf_counter()
        m.solve()
        t_solve = np.array([perf_counter() - tic])
        np.savetxt(BASE_PATH / f'ex{ex_id}_s{comm.size}_time.dat', t_solve, fmt='%.4e')


def problem1(T: float, neumann: bool):
    # 1D heat equation
    p = StationaryProblem(domain=LineDomain([0, 1], left='neumann', right='neumann') if neumann else LineDomain([0, 1]),
                          diffusion=ConstantFunction(1., 1),
                          rhs=ExpressionFunction('100*sin(5*pi*t[0])*(1+cos(3*pi*x[0]))', 1, parameters={'t': 1}),)
    pp = InstationaryProblem(p,
                             initial_data=ExpressionFunction('10*(0.6 <= x[0] <= 0.8)', 1),
                             T=T)
    m, disc_data = discretize_instationary_cg(pp, diameter=1/100, nt=100 * T)
    if not neumann:
        m = remove_dirichlet(m, disc_data['boundary_info'].dirichlet_boundaries(disc_data['grid'].dim))
    return m


def problem2(T: float, neumann: bool):
    # 1D time-dependent diffusion
    p = StationaryProblem(domain=LineDomain([0, 1], left='neumann', right='neumann') if neumann else LineDomain([0, 1]),
                          diffusion=ExpressionFunction('1 + 0.9*sin(7*pi*t[0]+2*pi*x[0])', 1, parameters={'t': 1}),
                          rhs=ExpressionFunction('100*sin(5*pi*t[0])*(1+cos(3*pi*x[0]))', 1, parameters={'t': 1}),)
    pp = InstationaryProblem(p,
                             initial_data=ExpressionFunction('10*(0.6 <= x[0] <= 0.8)', 1),
                             T=T)
    m, disc_data = discretize_instationary_cg(pp, diameter=1/100, nt=100 * T)
    if not neumann:
        m = remove_dirichlet(m, disc_data['boundary_info'].dirichlet_boundaries(disc_data['grid'].dim))
    return m


def problem3(T: float, neumann: bool):
    # 2D time-dependent diffusion
    p = StationaryProblem(domain=(RectDomain(left='neumann', right='neumann', top='neumann', bottom='neumann')
                                  if neumann else RectDomain()),
                          diffusion=ExpressionFunction('1.'
                                                       '+ 0.9*sin(7*pi*t[0])*(0 <= x[0] <= 0.5)*(0 <= x[1] <= 0.5)'
                                                       '+ 0.9*cos(5*pi*t[0])*(0.5 <= x[0] <= 1)*(0.5 <= x[1] <= 1)',
                                                       2, parameters={'t': 1}),
                          rhs=ExpressionFunction('100*sin(5*pi*t[0])*(1+cos(3*pi*x[0]))*(1+sin(4*pi*x[1]))',
                                                 2, parameters={'t': 1}),)
    pp = InstationaryProblem(p,
                             initial_data=ExpressionFunction('10*(0.6 <= x[0] <= 0.8)*(0.6 <= x[1] <= 0.8)', 2),
                             T=T)
    m, disc_data = discretize_instationary_cg(pp, diameter=1/100, nt=100 * T)
    if not neumann:
        m = remove_dirichlet(m, disc_data['boundary_info'].dirichlet_boundaries(disc_data['grid'].dim))
    return m


def problem4(small=False):
    m = heatsink_model(data_file='heatsink_small.vtu' if small else 'heatsink.vtu')
    return m


def experiment_1_dirichlet():
    def setup(k):
        m = problem1(2**k, neumann=False)
        return m, PararealNoCoarse(m)
    run_experiment(setup, 4, '1_dir', same_m=False)


def experiment_1_dirichlet_euler():
    def setup(k):
        m = problem1(2**k, neumann=False)
        return m, PararealEuler(m)
    run_experiment(setup, 4, '1_dir_euler', same_m=False)


def experiment_1_neumann():
    def setup(k):
        m = problem1(2**k, neumann=True)
        return m, PararealNoCoarse(m)
    run_experiment(setup, 4, '1_neu', same_m=False)


def experiment_1_neumann_euler():
    def setup(k):
        m = problem1(2**k, neumann=True)
        return m, PararealEuler(m)
    run_experiment(setup, 4, '1_euler', same_m=False)


def experiment_1_neumann_r1():
    def setup(k):
        m = problem1(2**k, neumann=True)
        modes = m.solution_space.ones()
        gram_schmidt(modes, copy=False, product=m.products['l2'])
        return m, PararealAPrioriCoarse(m, modes, product=m.products['l2'], homogeneous=True)
    run_experiment(setup, 4, '1_neu_r1', same_m=False)


def experiment_1_dirichlet_fourier():
    m = problem1(1, neumann=False)
    def setup(k):
        modes = m.solution_space.empty()
        for kk in range(1, k+1):
            modes.append(modes.space.from_numpy(np.sin(kk*np.pi*np.linspace(0, 1, 101)[1:-1])))
        gram_schmidt(modes, copy=False, product=m.products['l2'])
        return m, PararealAPrioriCoarse(m, modes, product=m.products['l2'], homogeneous=True)
    run_experiment(setup, 4, '1_dir_fou')


def experiment_2_neumann_fourier():
    m = problem2(1, neumann=True)
    def setup(k):
        modes = m.solution_space.empty()
        for kk in range(0, k):
            modes.append(modes.space.from_numpy(np.cos(kk*np.pi*np.linspace(0, 1, 101))))
        gram_schmidt(modes, copy=False, product=m.products['l2'])
        return m, PararealAPrioriCoarse(m, modes, product=m.products['l2'], homogeneous=False)
    run_experiment(setup, 6, '2_neu_fou')


small_jobs = [
    experiment_1_dirichlet,
    experiment_1_dirichlet_euler,
    experiment_1_neumann,
    experiment_1_neumann_euler,
    experiment_1_neumann_r1,
    experiment_1_dirichlet_fourier,

    experiment_2_neumann_fourier,
    ('2_neu',   run_euler_experiment,  (lambda: problem2(1, neumann=True),),     {}),
    ('2_neu',   run_svals_experiment,  (lambda: problem2(1, neumann=True), 30),  {}),
    ('2_neu',   run_pararb_experiment, (lambda: problem2(1, neumann=True), 6),   {'p': 0, 'q': 0}),
    ('2_neu',   run_pararb_experiment, (lambda: problem2(1, neumann=True), 6),   {'p': 1, 'q': 0}),
    ('2_neu',   run_pararb_experiment, (lambda: problem2(1, neumann=True), 6),   {'p': 2, 'q': 0}),
    ('2_neu',   run_pararb_experiment, (lambda: problem2(1, neumann=True), 6),   {'p': 3, 'q': 0}),
    ('2_neu',   run_pararb_experiment, (lambda: problem2(1, neumann=True), 6),   {'svd_method': 'exact'}),

    ('3',   run_solve_experiment,  (lambda: problem3(1, neumann=False),),    {}),
    ('3',   run_euler_experiment,  (lambda: problem3(1, neumann=False),),    {}),
    ('3',   run_svals_experiment,  (lambda: problem3(1, neumann=False), 30), {}),
    ('3',   run_pararb_experiment, (lambda: problem3(1, neumann=False), 20), {'p': 0, 'q': 0}),
    ('3',   run_pararb_experiment, (lambda: problem3(1, neumann=False), 20), {'p': 1, 'q': 0}),
    ('3',   run_pararb_experiment, (lambda: problem3(1, neumann=False), 20), {'p': 2, 'q': 0}),
    ('3',   run_pararb_experiment, (lambda: problem3(1, neumann=False), 20), {'p': 3, 'q': 0}),
    ('3',   run_pararb_experiment, (lambda: problem3(1, neumann=False), 20), {'svd_method': 'exact'}),
]


large_jobs = [
    ('4',       run_solve_experiment,  (lambda: problem4(),),                    {}),
    ('4',       run_pararb_experiment, (lambda: problem4(), 4),                  {'p': 0, 'q': 0}),
    ('4',       run_pararb_experiment, (lambda: problem4(), 4),                  {'p': 1, 'q': 0}),
    ('4',       run_euler_experiment,  (lambda: problem4(),),                    {}),
    ('4',       run_svals_experiment,  (lambda: problem4(), 10),                 {}),
]



def tqdm(*args, **kwargs):
    if rank == 0:
        return real_tqdm(*args, **kwargs)
    else:
        return args[0]


def run_jobs(jobs):
    for job in tqdm(jobs, 'total'):
        if isinstance(job, tuple):
            job[1](job[0], *job[2], **job[3])
        else:
            job()
        if rank == 0:
            print()
            print()


if __name__ == '__main__':
    assert len(sys.argv) == 2
    assert sys.argv[1] in {'small', 'large'}
    jobs = small_jobs if sys.argv[1] == 'small' else large_jobs
    run_jobs(jobs)
