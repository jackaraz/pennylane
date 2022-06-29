import pennylane as qml


class TransformedQfunc:
    """A transformed qfunc. Can be called with the same call signature as the original
    quantum function.  Call returns a ``Circuit`` object.

    Args:
        qfunc (qfunc, Circuit): Initial quantum function or circuit
        circuit_transform (function)

    Keyword Arguments:
        transform_args (tuple): any arguments to the circuit transform
        transform_kwargs (dict): any keyword arguments to the circuit transform

    """

    def __init__(self, qfunc, circuit_transform, transform_args=None, transform_kwargs=None):

        self.qfunc = qfunc
        self.circuit_transform = circuit_transform
        self.transform_args = tuple() if transform_args is None else transform_args
        self.transform_kwargs = dict() if transform_kwargs is None else transform_kwargs

    def __call__(self, *args, **kwargs):
        initial_circuit = qml.circuit.make_circuit(self.qfunc, *args, **kwargs)
        return self.circuit_transform(
            initial_circuit, *self.transform_args, **self.transform_kwargs
        )
