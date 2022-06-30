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
r"""
Contains the encoding of a Hamiltonian into a unitary.
"""
import numpy as np
import pennylane as qml
import pennylane.numpy as qnp
from pennylane.wires import Wires
from pennylane.operation import Operation, AnyWires

get_binary_lst = lambda x, n: [int(i) for i in format(x, 'b').zfill(n)]


class HamiltonainEncoding(Operation):

    num_wires = AnyWires
    grad_method = None

    def __init__(self, coeffs, ops, a_wires, h_wires, do_queue=True, id=None):
        r"""Given a list of coefficients, unitary ops and a series of wires,
        apply the hamiltonian encoding """
        wires = Wires(a_wires + h_wires)
        self._hyperparameters = {"a_wires": a_wires, "h_wires": h_wires}
        self.ops = ops

        num_terms = len(ops)
        assert np.log2(num_terms) <= len(a_wires)  # we have enough ancillary qubits to encode H

        super().__init__(coeffs, ops, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        """The coeffs ad ops that make up the operator"""
        return 2

    @staticmethod
    def compute_decomposition(coeffs, ops, wires, a_wires, h_wires):
        """Return a QubitUnitary representing the embedded hamiltonian."""
        num_ancilla = len(a_wires)
        num_wires = len(wires)

        final_unitary = qnp.zeros((2**num_wires, 2**num_wires))

        for i, op in enumerate(ops):
            project_basis_state = get_binary_lst(i, num_ancilla)
            tensor_op = qml.Projector(project_basis_state, wires=a_wires) @ op

            final_unitary += tensor_op.matrix()

        return [qml.QubitUnitary(final_unitary, wires=wires)]

    def queue(self, context=qml.QueuingContext):
        for op in self.ops:
            context.safe_update_info(op, owner=self)
            context.append(self, owns=op)

        return self
