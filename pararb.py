from time import perf_counter

import numpy as np
from mpi4py import MPI
from pymor.algorithms.rand_la import randomized_svd
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import AdjointOperator, ConstantOperator, LowRankOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.parameters.base import ParametricObject
from pymor.tools.random import new_rng


class TimeSlice(ParametricObject):

    def __init__(self, operator, mass, rhs, T0, T1, nt):
        assert operator.source == operator.range == mass.source == mass.range == rhs.range
        self.__auto_init(locals())
        self.dt = (self.T1 - self.T0)/self.nt
        self.solution_space = operator.source
        self.parameters_internal = {'t': 1}
        op = self.mass + self.dt*self.operator
        if not op.parametric:
            op = op.assemble()
            adjoint_op = op.H.assemble()
        else:
            adjoint_op = AdjointOperator(op)  # necessary, since AdjointOperator does not implement assemble
                                              # and LincombOperator implements .H, causing
                                              # apply_inverse to fail to assemble a matrix
        self._time_step_op, self._adjoint_time_step_op, self._inhomogeneous_part = op, adjoint_op, None

    @classmethod
    def from_model(cls, model, comm=MPI.COMM_WORLD):
        assert isinstance(model, InstationaryModel)
        rank = comm.Get_rank()
        num_slices = comm.size

        nt = model.time_stepper.nt // num_slices
        assert model.time_stepper.nt == nt * num_slices

        Dt = model.T / num_slices
        T0 = Dt * rank
        T1 = Dt * (rank+1)

        return cls(model.operator, model.mass, model.rhs, T0, T1, nt)

    def solve(self, initial_data, mu=None, homogeneous=False, only_final_time=False):
        if mu is not None:
            raise NotImplementedError
        mu = self.parameters.parse(mu)

        compute_hom = initial_data is not None
        compute_inhom = self._inhomogeneous_part is None

        assert compute_hom or compute_inhom

        if compute_hom:
            U = initial_data
            if not only_final_time:
                UU = self.solution_space.empty(reserve=self.nt)

        if compute_inhom:
            U_inhom = self.solution_space.zeros()

        op = self._time_step_op
        t = self.T0
        for _ in range(self.nt):
            t += self.dt
            mu = mu.with_(t = t)
            op_assembled = op.assemble(mu)

            if not homogeneous or compute_inhom:
                inhom_rhs = self.rhs.as_vector(mu) * self.dt

            if compute_hom:
                rhs = self.mass.apply(U)
                if not homogeneous:
                    rhs += inhom_rhs
                U = op_assembled.apply_inverse(rhs)
                if not only_final_time:
                    UU.append(U)

            if compute_inhom:
                rhs = self.mass.apply(U_inhom)
                rhs += inhom_rhs
                U_inhom = op_assembled.apply_inverse(rhs)

        if compute_inhom:
            self._inhomogeneous_part = U_inhom

        if not compute_hom:
            return
        if only_final_time:
            return U
        else:
            return UU

    def solve_adjoint(self, terminal_data, mu=None, only_final_time=False):
        if mu is not None:
            raise NotImplementedError
        mu = self.parameters.parse(mu)
        adjoint_op = self._adjoint_time_step_op
        if not only_final_time:
            UU = self.solution_space.empty(reserve=self.nt)
        U = terminal_data
        t = self.T1
        for _ in range(self.nt):
            mu = mu.with_(t=t)
            U = self.mass.apply(adjoint_op.apply_inverse(U, mu=mu))
            if not only_final_time:
                UU.append(U)
            t -= self.dt
        if only_final_time:
            return U
        else:
            return UU

    @property
    def transfer_operator(self):
        return TransferOperator(self)

    @property
    def inhomogeneous_part(self):
        if self.parametric:
            raise NotImplementedError
        if self._inhomogeneous_part is None:
            self.solve(None)
        return self._inhomogeneous_part


class TransferOperator(Operator):

    linear = True

    def __init__(self, ts):
        self.ts = ts
        self.source = self.range = self.ts.solution_space

    def apply(self, U, mu=None):
        return self.ts.solve(U, mu=mu, homogeneous=True, only_final_time=True)

    def apply_adjoint(self, V, mu=None):
        return self.ts.solve_adjoint(V, mu=mu, only_final_time=True)


class Parareal:

    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm

    def get_F(self):
        raise NotImplementedError

    def get_G(self):
        raise NotImplementedError

    def run(self, U0, it=None):
        rank = self.comm.Get_rank()
        it = it or self.comm.size
        times = np.empty(it+1)

        # setup
        tic = perf_counter()
        F = self.get_F()
        G = self.get_G()
        times[0] = perf_counter()

        # main loop
        to_send = U0.zeros()
        Us, Ucont = [], []
        for i in range(it):
            # receive new initial data
            if rank == 0:
                U = U0
            else:
                U = np.empty(U0.dim)
                self.comm.Recv(U, source=rank-1)
                U = U0.space.from_numpy(U)

            Us.append(U)

            # coarse update
            GU = G(U)
            to_send += GU

            # send new initial data
            if rank+1 < self.comm.size:
                self.comm.Send(to_send.to_numpy(), dest=rank+1)

            # fine update
            FU = F(U)
            Ucont.append(FU)

            times[i+1] = perf_counter()
            if i+1 < it:
                # begin coarse update for next iteration
                to_send = FU[-1] - GU

        times -= tic
        return Us, Ucont, times


class PararealForModel(Parareal):

    def __init__(self, model, comm=MPI.COMM_WORLD):
        super().__init__(comm)
        self.ts = TimeSlice.from_model(model, comm=comm)
        self.U0 = model.initial_data.as_range_array()

    def get_F(self):
        return lambda U: self.ts.solve(U)

    def run(self, it=None):
        return super().run(self.U0, it)


class PararealNoCoarse(PararealForModel):
    def get_G(self):
        return lambda U: U.zeros()


class PararealEuler(PararealForModel):

    def get_G(self):
        G_ts = self.ts.with_(nt=1)
        return lambda U: G_ts.solve(U)[-1]


class LowRankParareal(PararealForModel):
    def __init__(self, model, product=None, homogeneous=False, comm=MPI.COMM_WORLD):
        super().__init__(model, comm=comm)
        self.product, self.homogeneous = product, homogeneous

    def get_G(self):
        U, S, V = self.get_U_S_V()
        if U:
            if self.product is not None:
                V = self.product.apply(V)  # LowRankOperator does not support products
            linear_part = LowRankOperator(U, np.diag(S), V)
        else:
            linear_part = ZeroOperator(self.ts.solution_space, self.ts.solution_space)
        if self.homogeneous:
            op = linear_part
        else:
            op = ConstantOperator(self.ts.inhomogeneous_part, self.ts.solution_space) + linear_part
        return lambda U: op.apply(U)


class PararealAPrioriCoarse(LowRankParareal):
    def __init__(self, model, basis, product=None, homogeneous=False, comm=MPI.COMM_WORLD):
        super().__init__(model, product=product, homogeneous=homogeneous, comm=comm)
        self.basis = basis

    def get_U_S_V(self):
        if len(self.basis) == 0:
            return None, None, None
        image = self.ts.transfer_operator.apply(self.basis)
        return image, np.ones(len(self.basis)), self.basis


class ParaRB(LowRankParareal):
    def __init__(self, model, modes, product=None, homogeneous=False, svd_method='rand',
                 p=None, q=None, comm=MPI.COMM_WORLD):
        assert svd_method in {'rand', 'exact'}
        super().__init__(model, product=product, homogeneous=homogeneous, comm=comm)
        self.modes, self.svd_method, self.p, self.q = modes, svd_method, p, q

    def get_U_S_V(self):
        if self.svd_method == 'exact':
            if self.modes == 0:
                return None, None, None
            U, S, V = exact_svd(self.ts.transfer_operator, num_svals=self.modes, product=self.product)
            self.svals = S
            return U, S, V
        elif self.svd_method == 'rand':
            # don't run the svd if modes == 0 and p == 0
            # if modes == 0 and p > 0, run the svd to get an error estimate
            if self.modes + self.p == 0:
                return None, None, None
            with new_rng():
                # don't use oversampling parameter and truncate outside to obtain additional
                # singular values for error estimation
                U, S, V = randomized_svd(
                    self.ts.transfer_operator, n=self.modes+self.p,
                    source_product=self.product, range_product=self.product,
                    oversampling=0, power_iterations=self.q
                )
            self.svals = S
            if self.modes == 0:
                return None, None, None
            else:
                return U[:self.modes], S[:self.modes], V[:self.modes]
        else:
            assert False


def exact_svd(op, product=None, num_svals=10):
    if product is not None:
        raise NotImplementedError
    from scipy.sparse.linalg import LinearOperator, svds
    scipy_op = LinearOperator((op.source.dim,)*2,
                              matvec=lambda U: op.apply(op.source.from_numpy(U.T)).to_numpy().T,
                              rmatvec=lambda V: op.apply_adjoint(op.range.from_numpy(V.T)).to_numpy().T,)
    U, s, V = svds(scipy_op, num_svals, which='LM', random_state=42)
    U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
    U = op.range.from_numpy(U.T)
    V = op.source.from_numpy(V)
    return U, s, V
