# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This submodule defines the symbolic operator that represents e raised to an operator.
"""
from scipy.linalg import expm
from scipy.sparse.linalg import expm as sparse_expm


from pennylane import math
from pennylane.operation import expand_matrix, Tensor
from pennylane.wires import Wires

from .symbolicop import SymbolicOp


class Exp(SymbolicOp):
    """Represents the exponential of an operator multiplied by a coefficient.

    Args:
        base (Operator):
        coeff (Number): A scalar coefficient of the operator

    **Example:**

    The rotation gates can be reproduced using the ``Exp`` operator:

    >>> base = qml.PauliX(0)
    >>> phi = 1.234
    >>> rx = Exp(base, -0.5j * phi)
    >>> qml.matrix(rx)
    array([[0.8156179+0.j        , 0.       -0.57859091j],
        [0.       -0.57859091j, 0.8156179+0.j        ]])
    >>> qml.matrix(qml.RX(phi, 0))
    array([[0.8156179+0.j        , 0.       -0.57859091j],
        [0.       -0.57859091j, 0.8156179+0.j        ]])

    If the coefficient is purely imaginary and the base operator is Hermitian, then
    the gate can be used in a circuit.

    """

    coeff = 1
    """The numerical coefficient of the operator in the exponential."""

    control_wires = Wires([])

    def __init__(self, base=None, coeff=1, do_queue=True, id=None):
        self.coeff = coeff
        super().__init__(base, do_queue=do_queue, id=id)
        if isinstance(base, Tensor):
            self._name = f"Exp({coeff} {base})"
        else:
            self._name = f"Exp({coeff} {base.name})"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def data(self):
        return [self.coeff, self.base.data]

    @data.setter
    def data(self, new_data):
        self.coeff = new_data[0]
        self.base.data = new_data[1]

    @property
    def num_params(self):
        return self.base.num_params + 1

    def matrix(self, wire_order=None):
        mat = expm(self.coeff * self.base.matrix())

        if wire_order is None or self.wires == Wires(wire_order):
            return mat

        return expand_matrix(mat, wires=self.wires, wire_order=wire_order)

    def sparse_matrix(self, wire_order=None):
        if wire_order is not None:
            raise NotImplementedError("Wire order is not implemented for sparse_matrix")

        base_smat = self.coeff * self.base.sparse_matrix()
        return sparse_expm(base_smat)

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def eigvals(self):
        return math.exp(self.base.eigvals())

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "Exp"

    def pow(self, z):
        return Exp(self.base, self.coeff * z)

    @property
    def is_hermitian(self):
        return self.base.is_hermitian and math.imag(self.coeff) == 0

    @property
    def _queue_category(self):
        if self.base.is_hermitian and math.real(self.coeff) == 0:
            return "_ops"
        return None
