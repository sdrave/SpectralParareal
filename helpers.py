import numpy as np
from pymor.algorithms.rules import RuleTable, match_always, match_class
from pymor.core.base import ImmutableObject
from pymor.models.interface import Model
from pymor.operators.constructions import LincombOperator, VectorOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


def remove_dirichlet(model, dirichlet_dofs):
    new_space = NumpyVectorSpace(model.solution_space.dim - len(dirichlet_dofs), id=model.solution_space.id)
    mask = np.full(model.solution_space.dim, True)
    mask[dirichlet_dofs] = False
    non_dirichlet_dofs = np.where(mask)[0]
    new_model = RemoveDirichletRules(model.solution_space, new_space, non_dirichlet_dofs).apply(model)
    new_visualizer = RemoveDirichletVisualizer(new_model.visualizer, model.solution_space, non_dirichlet_dofs)
    return new_model.with_(visualizer=new_visualizer)


class RemoveDirichletVisualizer(ImmutableObject):

    def __init__(self, orig, affected_space, non_dirichlet_dofs):
        self.__auto_init(locals())

    def add_dirichlet_dofs(self, U):
        data = np.zeros((len(U), self.affected_space.dim))
        data[:, self.non_dirichlet_dofs] = U.to_numpy()
        return self.affected_space.from_numpy(data)

    def visualize(self, U, **kwargs):
        assert isinstance(U, VectorArray) \
            or (isinstance(U, tuple)
                and all(isinstance(u, VectorArray) for u in U)
                and all(len(u) == len(U[0]) for u in U))
        if isinstance(U, VectorArray):
            U = (U,)

        U = tuple(self.add_dirichlet_dofs(u) for u in U)
        self.orig.visualize(U, **kwargs)


class RemoveDirichletRules(RuleTable):
    """|RuleTable| for the :func:`preassemble` algorithm."""

    def __init__(self, affected_space, new_space, non_dirichlet_dofs):
        super().__init__(use_caching=True)
        self.__auto_init(locals())

    @match_class(Model, LincombOperator)
    def action_recurse(self, op):
        return self.replace_children(op)

    @match_class(NumpyMatrixOperator)
    def action_numpy_matrix_operator(self, op):
        if op.range == self.affected_space:
            if op.source == self.affected_space:
                mat = op.matrix[:, self.non_dirichlet_dofs][self.non_dirichlet_dofs, :]
            else:
                mat = op.matrix[self.non_dirichlet_dofs, :]
        else:
            raise NotImplementedError
        return op.with_(matrix=mat)

    @match_class(VectorOperator)
    def action_vector_operator(self, op):
        vec = op.vector.to_numpy()[:,self.non_dirichlet_dofs]
        return op.with_(vector=self.new_space.from_numpy(vec))

    @match_class(ZeroOperator)
    def action_zero_operator(self, op):
        return ZeroOperator(range=self.new_space if op.range == self.affected_space else op.range,
                            source=self.new_space if op.source == self.affected_space else op.source)

    @match_always
    def action_generic_wrapper(self, op):
        return RemoveDirichletOperator(op, self.affected_space, self.new_space, self.non_dirichlet_dofs)


class RemoveDirichletOperator(Operator):

    def __init__(self, op, affected_space, new_space, non_dirichlet_dofs):
        self.__auto_init(locals())
        self.source = new_space if op.source == affected_space else op.source
        self.range = new_space if op.range == affected_space else op.range
        self.linear = op.linear

    def _affected_to_new(self, U):
        return self.new_space.from_numpy(U.to_numpy()[:, self.non_dirichlet_dofs])

    def _new_to_affected(self, U):
        V = self.affected_space.zeros(len(U))
        V.to_numpy()[:, self.non_dirichlet_dofs] = U.to_numpy()
        return V

    def _source_to_orig(self, U):
        return U if self.source == self.op.source else self._new_to_affected(U)

    def _orig_to_source(self, U):
        return U if self.source == self.op.source else self._affected_to_new(U)

    def _range_to_orig(self, U):
        return U if self.range == self.op.range else self._new_to_affected(U)

    def _orig_to_range(self, U):
        return U if self.range == self.op.range else self._affected_to_new(U)

    def apply(self, U, mu=None):
        return self._orig_to_range(self.op.apply(self._source_to_orig(U), mu=mu))

    def assemble(self, mu=None):
        op = self.op.assemble(mu)
        return RemoveDirichletRules(self.affected_space, self.new_space, self.non_dirichlet_dofs).apply(op)
