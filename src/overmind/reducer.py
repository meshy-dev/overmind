# -*- coding: utf-8 -*-

# -- stdlib --
from multiprocessing.reduction import ForkingPickler

# -- third party --
# -- own --
from .shmem import SharedMemory
from overmind.server import OvermindService


# -- code --
current_service: OvermindService | None = None


class SharedMemoryView:
    # client memoryview -> server WrappedMemoryView (via reduce_memoryview_on_client)
    # server WrappedMemoryView -> client memoryview (via reduce_memoryview_for_server)
    def __init__(self, data: bytes | memoryview):
        self._shmem = SharedMemory.create(len(data))
        assert self._shmem.buf
        self._shmem.buf[:] = data


def rebuild_memoryview_on_client(v: bytes | SharedMemoryView):
    if isinstance(v, bytes):
        return memoryview(v)
    elif isinstance(v, SharedMemoryView):
        return v._shmem.buf.toreadonly()
    else:
        raise Exception(f'Bad v: {v}')


def rebuild_memoryview_on_server(v: bytes):
    print('rebuild server', v)
    return memoryview(v)


def reduce_memoryview_on_client(v: memoryview):
    return (rebuild_memoryview_on_server, (bytes(v),))


def reduce_memoryview_on_server(v: memoryview):
    assert current_service

    if len(v) < 1 * 1024 * 1024:
        return (rebuild_memoryview_on_client, (bytes(v),))

    if id(v) in current_service._tracked_memoryviews:
        wrapped = current_service._tracked_wrapped[id(v)]
    else:
        current_service._tracked_memoryviews[id(v)] = v
        current_service._tracked_wrapped[id(v)] = wrapped = SharedMemoryView(v)

    return (rebuild_memoryview_on_client, (wrapped,))


def init_reductions_client():
    ForkingPickler.register(memoryview, reduce_memoryview_on_client)


def init_reductions_server():
    ForkingPickler.register(memoryview, reduce_memoryview_on_server)
