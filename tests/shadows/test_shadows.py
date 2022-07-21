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
Tests for classical shadows
"""
import pytest
import pennylane as qml
import pennylane.numpy as np
from pennylane.shadows import ClassicalShadow

np.random.seed(777)

wires = range(5)
shots = 10000
dev = qml.device("default.qubit", wires=wires, shots=shots)
@qml.qnode(dev)
def qnode(n_wires):
    for i in range(n_wires):
        qml.Hadamard(i)
    return qml.classical_shadow(wires=range(n_wires))

shadows = [ClassicalShadow(*qnode(n_wires)) for n_wires in range(2, 5)]

class TestUnitTestClassicalShadows:
    """Unit Tests for ClassicalShadow class"""

    @pytest.mark.parametrize("shadow", shadows)
    def test_unittest_local_snapshots(self, shadow):
        """Test the output shape of local_snapshots method"""
        T, n = shadow.bitstrings.shape
        assert (T, n) == shadow.recipes.shape
        assert shadow.local_snapshots().shape == (T, n, 2, 2)

    @pytest.mark.parametrize("shadow", shadows)
    def test_unittest_global_snapshots(self, shadow):
        """Test the output shape of global_snapshots method"""
        T, n = shadow.bitstrings.shape
        assert (T, n) == shadow.recipes.shape
        assert shadow.global_snapshots().shape == (T, 2**n, 2**n)


@pytest.mark.parametrize("shadow", shadows)
def test_pauli_string_expval(shadow):
    """Testing the output of expectation values match those of exact evaluation"""

    o1 = qml.PauliX(0)
    res1 = shadow.expval_observable(o1, k=2)

    o2 = qml.PauliX(0) @ qml.PauliX(1)
    res2 = shadow.expval_observable(o1, k=2)

    res_exact = 1.
    assert qml.math.allclose(res1, res_exact, atol=1e-1)
    assert qml.math.allclose(res2, res_exact, atol=1e-1)

# def test_expval_H():
#     """Testing the output of expectation values match those of exact evaluation"""