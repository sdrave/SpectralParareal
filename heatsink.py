import meshio
import numpy as np
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.core.base import ImmutableObject
from pymor.core.logger import getLogger
from pymor.models.basic import InstationaryModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace
from skfem import Basis, BilinearForm, ElementTetP0, ElementTetP1, Mesh
from skfem.helpers import dot, grad
from skfem.io.meshio import to_meshio
from skfem.models.poisson import mass, unit_load


def heatsink_model(data_file='heatsink.vtu', fin_conductivity=10., pipe_conductivity=1000., base_conductivity=100.):
    logger = getLogger('heatsink')

    logger.info('Loading mesh.')
    mesh = Mesh.load('heatsink_data/' + data_file)
    factor = 1/np.max(np.max(mesh.doflocs, axis=1) - np.min(mesh.doflocs, axis=1))
    mesh = mesh.scaled(factor)

    logger.info('Bulding finite element spaces.')
    Vh = Basis(mesh, ElementTetP1())
    Vh0 = Vh.with_element(ElementTetP0())

    logger.info('Assembling ...')
    bottom_boundary = mesh.boundaries['Bottom_Surface']
    pipe_boundary = np.intersect1d(mesh.facets_around('Pipes_Volume'), mesh.boundary_facets())
    base_boundary = np.setdiff1d(np.intersect1d(mesh.facets_around('BaseBody_Volume'), mesh.boundary_facets()),
                                 bottom_boundary)
    fin_boundary = np.setdiff1d(np.setdiff1d(np.setdiff1d(mesh.boundary_facets(), base_boundary), pipe_boundary),
                                bottom_boundary)

    conductivity = Vh0.zeros()
    conductivity[:] = fin_conductivity
    conductivity[Vh0.get_dofs(elements='Pipes_Volume')] = pipe_conductivity
    conductivity[Vh0.get_dofs(elements='BaseBody_Volume')] = base_conductivity

    @BilinearForm
    def a(u, v, w):
        return dot(w['conductivity'] * grad(u), grad(v))

    A = a.assemble(Vh, conductivity=Vh0.interpolate(conductivity))
    R = mass.assemble(Vh.boundary(fin_boundary))
    M = mass.assemble(Vh)
    b = unit_load.assemble(Vh.boundary(bottom_boundary))

    model = InstationaryModel(
        initial_data=NumpyVectorSpace.from_numpy(np.zeros(M.shape[0])),
        mass=NumpyMatrixOperator(M),
        operator=NumpyMatrixOperator(A) + NumpyMatrixOperator(R) * ExpressionParameterFunctional('0.5 + t', {'t': 1}),
        rhs=(NumpyMatrixOperator(b.reshape((-1, 1)))
             * ExpressionParameterFunctional(
                '(t[0]*50/0.3)*(t[0] <= 0.3)'
                ' + 50*(1 + sign(sin((t[0]-0.3)/0.3*4*2*pi)))*(0.3 < t[0] <= 0.6)'
                ' + 50*(1 + cos((t[0]-0.6)/0.4*10*2*pi))*(0.6 < t[0])',
                {'t': 1},
               )
            ),
        T=1.,
        time_stepper=ImplicitEulerTimeStepper(500),
        visualizer=Visualizer(mesh)
    )
    return model


class Visualizer(ImmutableObject):
    def __init__(self, skfem_mesh):
        self.__auto_init(locals())
        self._meshio_mesh = None

    def visualize(self, U, path='./out.xdmf', label='solution', **kwargs):
        if self._meshio_mesh is None:
            self._meshio_mesh = to_meshio(self.skfem_mesh)
        U = U.to_numpy()
        mesh = self._meshio_mesh
        with meshio.xdmf.TimeSeriesWriter(path) as writer:
            writer.write_points_cells(mesh.points, mesh.cells)
            for i, u in enumerate(U):
                writer.write_data(i, point_data={label: u})


if __name__ == '__main__':
    from pymor.core.logger import set_log_levels
    set_log_levels({'heatsink': 'INFO'})
    model = heatsink_model()
    U = model.solve()
    model.visualize(U)
    print(f'FOM dimension: {U.dim}')
