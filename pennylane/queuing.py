# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from threading import RLock
from contextlib import contextmanager


class QueueManager:

    _recording_queues = []

    @classmethod
    def add_recording_queue(cls, queue):
        cls._recording_queues.append(queue)

    @classmethod
    def remove_recording_queue(cls):
        return cls._recording_queues.pop()

    @classmethod
    def recording(cls):
        return bool(cls._recording_queues)

    @classmethod
    def active_queue(cls):
        return cls._recording_queues[-1] if cls.recording() else None

    @classmethod
    @contextmanager
    def stop_recording(cls):
        active_contexts = cls._recording_queues
        cls._recording_queues = []
        yield
        cls._recording_queues = active_contexts

    @classmethod
    def append(cls, obj, **kwargs):
        if cls.recording():
            cls.active_queue().append(obj, **kwargs)

    @classmethod
    def remove(cls, obj):
        if cls.recording():
            cls.active_queue().remove(obj)

    @classmethod
    def update_info(cls, obj, **kwargs):
        if cls.recording():
            cls.active_queue().update_info(obj, **kwargs)

    @classmethod
    def get_info(cls, obj):
        if cls.recording():
            return cls.active_queue().get_info(obj)


class Queue:

    _lock = RLock()

    def __init__(self, do_queue=True):
        self._queue = OrderedDict()
        self.do_queue = do_queue

    def append(self, obj, **kwargs):
        self._queue[obj] = kwargs

    def remove(self, obj):
        del self._queue[obj]

    def update_info(self, obj, **kwargs):
        if obj in self._queue:
            self._queue[obj].update(kwargs)

    def get_info(self, obj):
        return self._queue.get(obj, {})

    @property
    def queue(self):
        return list(self._queue)

    def __enter__(self):
        Queue._lock.acquire()
        try:
            if self.do_queue and QueueManager.recording():
                QueueManager.active_queue().append(self)
            QueueManager.add_recording_queue(self)
            return self
        except Exception as _:
            Queue._lock.release()
            raise

    def __exit__(self, exception_type, exception_value, traceback):
        QueueManager.remove_recording_queue()

    @staticmethod
    @contextmanager
    def stop_recording():
        active_contexts = QueueManager._recording_queues
        QueueManager._recording_queues = []
        yield
        QueueManager._recording_queues = active_contexts

    def __iter__(self):
        return iter(self.queue)

    def items(self):
        return self._queue.items()

    def __getitem__(self, idx):
        return self.queue[idx]

    def __len__(self):
        return len(self.queue)


def process_queue(queue: Queue):
    list_dict = {"_ops": [], "_measurements": []}
    list_order = {"_ops": 0, "_measurements": 1}
    current_list = "_ops"

    for obj, info in queue.items():

        if "owner" not in info and getattr(obj, "_queue_category", None) is not None:

            if list_order[obj._queue_category] > list_order[current_list]:
                current_list = obj._queue_category
            elif list_order[obj._queue_category] < list_order[current_list]:
                raise ValueError(
                    f"{obj._queue_category[1:]} operation {obj} must occur prior "
                    f"to {current_list[1:]}. Please place earlier in the queue."
                )
            list_dict[obj._queue_category].append(obj)

    return tuple(list_dict["_ops"]), tuple(list_dict["_measurements"])


def apply(op, context=QueueManager):
    op.__copy__().queue()
