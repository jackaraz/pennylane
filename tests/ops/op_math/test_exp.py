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

import pytest
from copy import copy

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.op_math import Exp


class TestInitialization:
    """Test the initalization process and standard properties."""

    def test_pauli_base(self):

        base = qml.PauliX("a")

        op = Exp(base, id="something")

        assert op.base is base
        assert op.coeff == 1
        assert op.name == "Exp"
        assert op.id == "something"

        assert op.num_params == 1
        assert op.parameters == [1, []]
        assert op.data == [1, []]

        assert op.wires == qml.wires.Wires("a")

    def test_provided_coeff(self):

        base = qml.PauliZ("b") @ qml.PauliZ("c")
        coeff = np.array(1.234)

        op = Exp(base, coeff)

        assert op.base is base
        assert op.coeff is coeff
        assert op.name == "Exp"

        assert op.num_params == 1
        assert op.parameters == [coeff, []]
        assert op.data == [coeff, []]

        assert op.wires == qml.wires.Wires(("b", "c"))

    def test_parametric_base(self):

        base_coeff = 1.23
        base = qml.RX(base_coeff, wires=5)
        coeff = np.array(-2.0)

        op = Exp(base, coeff)

        assert op.base is base
        assert op.coeff is coeff
        assert op.name == "Exp"

        assert op.num_params == 2
        assert op.data == [coeff, [base_coeff]]

        assert op.wires == qml.wires.Wires(5)


class TestProperties:
    def test_data(self):

        phi = np.array(1.234)
        coeff = np.array(2.345)

        base = qml.RX(phi, wires=0)
        op = Exp(base, coeff)

        assert op.data == [coeff, [phi]]

        new_data = [-2.1, [-3.4]]
        op.data = new_data

        assert op.data == new_data
        assert op.coeff == -2.1
        assert base.data == [-3.4]

    def test_queue_category_ops(self):
        assert Exp(qml.PauliX(0), -1.234j)._queue_category == "_ops"

        assert Exp(qml.PauliX(0), 1 + 2j)._queue_category is None

        assert Exp(qml.RX(1.2, 0), -1.2j)._queue_category is None

    def test_is_hermitian(self):
        assert Exp(qml.PauliX(0), -1.0).is_hermitian

        assert not Exp(qml.PauliX(0), 1.0 + 2j).is_hermitian

        assert not Exp(qml.RX(1.2, wires=0)).is_hermitian


class TestMatrix:
    def test_matrix_rx(self):
        """Test the matrix comparing to the rx gate."""
        phi = np.array(1.234)
        exp_rx = Exp(qml.PauliX(0), -0.5j * phi)
        rx = qml.RX(phi, 0)

        assert qml.math.allclose(exp_rx.matrix(), rx.matrix())

    def test_sparse_matrix(self):
        """Test the sparse matrix function."""
        from scipy.sparse import csr_matrix

        H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        H = csr_matrix(H)
        base = qml.SparseHamiltonian(H, wires=0)

        op = Exp(base, 3)

        sp_format = "lil"
        sparse_mat = op.sparse_matrix(format=sp_format)
        assert sparse_mat.format == sp_format

        dense_mat = qml.matrix(op)

        assert qml.math.allclose(sparse_mat.toarray(), dense_mat)


class TestMiscMethods:
    def test_diagonalizing_gates(self):

        base = qml.PauliX(0)
        op = Exp(base, 1 + 2j)
        for op1, op2 in zip(base.diagonalizing_gates(), op.diagonalizing_gates()):
            assert qml.equal(op1, op2)

    def test_pow(self):
        """Test the pow decomposition method."""
        base = qml.PauliX(0)
        coeff = 2j
        z = 0.3

        op = Exp(base, coeff)
        pow_op = op.pow(z)

        assert isinstance(pow_op, Exp)
        assert pow_op.base is base
        assert pow_op.coeff == coeff * z

    def test_label(self):

        op = Exp(qml.PauliZ(0), 2 + 3j)
        assert op.label(decimals=4) == "Exp"


class TestIntegration:
    """Test Exp with gradients in qnodes."""

    @pytest.mark.jax
    def test_jax_qnode(self):
        """Test the execution and gradient of a jax qnode."""

        import jax
        from jax import numpy as jnp

        phi = jnp.array(1.234)

        @qml.qnode(qml.device("default.qubit.jax", wires=1), interface="jax")
        def circ(phi):
            Exp(qml.PauliX(0), -0.5j * phi)
            return qml.expval(qml.PauliZ(0))

        res = circ(phi)
        assert qml.math.allclose(res, jnp.cos(phi))
        grad = jax.grad(circ)(phi)
        assert qml.math.allclose(grad, -jnp.sin(phi))

    @pytest.mark.tf
    def test_tensorflow_qnode(self):
        """test the execution of a tensorflow qnode."""
        import tensorflow as tf

        phi = tf.Variable(1.2, dtype=tf.complex128)

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tensorflow")
        def circ(phi):
            Exp(qml.PauliX(0), -0.5j * phi)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circ(phi)

        phi_grad = tape.gradient(res, phi)

        assert qml.math.allclose(res, tf.cos(phi))
        assert qml.math.allclose(phi_grad, -tf.sin(phi))

    @pytest.mark.torch
    def test_torch_qnode(self):
        """Test execution with torch."""
        import torch

        phi = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch")
        def circuit(phi):
            Exp(qml.PauliX(0), -0.5j * phi)
            return qml.expval(qml.PauliZ(0))

        res = circuit(phi)
        assert qml.math.allclose(res, torch.cos(phi))

        res.backward()
        assert qml.math.allclose(phi.grad, -torch.sin(phi))

    @pytest.mark.autograd
    def test_autograd_qnode(self):
        """Test execution and gradient with pennylane numpy array."""
        phi = qml.numpy.array(1.2)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(phi):
            Exp(qml.PauliX(0), -0.5j * phi)
            return qml.expval(qml.PauliZ(0))

        res = circuit(phi)
        assert qml.math.allclose(res, qml.numpy.cos(phi))

        grad = qml.grad(circuit)(phi)
        assert qml.math.allclose(grad, -qml.numpy.sin(phi))

    @pytest.mark.autograd
    def test_autograd_measurement(self):
        """Test exp in a measurement with gradient and autograd."""

        x = qml.numpy.array(2)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit(x):
            qml.Hadamard(0)
            return qml.expval(Exp(qml.PauliZ(0), x))

        res = circuit(x)
        expected = 0.5 * (np.exp(x) + np.exp(-x))
        assert qml.math.allclose(res, expected)

        grad = qml.grad(circuit)(x)
        expected_grad = 0.5 * (np.exp(x) - np.exp(-x))
        assert qml.math.allclose(grad, expected_grad)

    @pytest.mark.torch
    def test_torch_measurement(self):
        """Test Exp in a measurement with gradient and torch."""

        import torch

        x = torch.tensor(2.0, requires_grad=True, dtype=float)

        @qml.qnode(qml.device("default.qubit", wires=1), interface="torch")
        def circuit(x):
            qml.Hadamard(0)
            return qml.expval(Exp(qml.PauliZ(0), x))

        res = circuit(x)
        expected = 0.5 * (torch.exp(x) + torch.exp(-x))
        assert qml.math.allclose(res, expected)

        res.backward()
        expected_grad = 0.5 * (torch.exp(x) - torch.exp(-x))
        assert qml.math.allclose(x.grad, expected_grad)

    @pytest.mark.jax
    def test_jax_measurement(self):
        """Test Exp in a measurement with gradient and jax."""

        import jax
        from jax import numpy as jnp

        x = jnp.array(2.0)

        @qml.qnode(qml.device("default.qubit", wires=1), interface="jax")
        def circuit(x):
            qml.Hadamard(0)
            return qml.expval(Exp(qml.PauliZ(0), x))

        res = circuit(x)
        expected = 0.5 * (jnp.exp(x) + jnp.exp(-x))
        assert qml.math.allclose(res, expected)

        grad = jax.grad(circuit)(x)
        expected_grad = 0.5 * (jnp.exp(x) - jnp.exp(-x))
        assert qml.math.allclose(grad, expected_grad)

    @pytest.mark.tf
    def test_tf_measurement(self):
        """Test Exp in a measurement with gradient and tensorflow."""
        import tensorflow as tf

        x = tf.Variable(2.0)

        @qml.qnode(qml.device("default.qubit", wires=1), interface="tensorflow")
        def circuit(x):
            qml.Hadamard(0)
            return qml.expval(Exp(qml.PauliZ(0), x))

        with tf.GradientTape() as tape:
            res = circuit(x)

        expected = 0.5 * (tf.exp(x) + tf.exp(-x))
        assert qml.math.allclose(res, expected)

        x_grad = tape.gradient(res, x)
        expected_grad = 0.5 * (tf.exp(x) - tf.exp(-x))
        assert qml.math.allclose(x_grad, expected_grad)
