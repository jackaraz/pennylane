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
This module contains the base quantum tape.
"""

import pennylane as qml
from pennylane.circuit import expand_circuit as expand_tape
from pennylane.circuit import TapeError


def get_active_tape():
    return qml.queuing.QueueManager.active_queue()


class QuantumTape(qml.queuing.AnnotatedQueue, qml.Circuit):
    def __init__(self, do_queue=False):
        qml.queuing.AnnotatedQueue.__init__(self, do_queue=do_queue)
        qml.Circuit.__init__(self, tuple(), tuple())
        self._observables = None

    def __exit__(self, exception_type, exception_value, traceback):
        qml.queuing.QueueManager.remove_recording_queue()
        self._observables = None
        self._process_queue()
        self._update()

    def _process_queue(self):
        self._ops, self._measurements = qml.queuing.process_queue(self._queue)

    __iter__ = qml.Circuit.__iter__
    __getitem__ = qml.Circuit.__getitem__
    __len__ = qml.Circuit.__len__
