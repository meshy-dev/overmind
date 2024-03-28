# -*- coding: utf-8 -*-

# -- stdlib --
from multiprocessing.reduction import ForkingPickler

# -- third party --
# -- own --
from .shmem import SharedMemory


# -- code --
current_service = None


class RebuildTorchModuleOnClient:

    def __init__(self, model, state_dev_map):
        self.model = model
        self.state_dev_map = state_dev_map

    def __reduce__(self):
        return (
            self._rebuild,
            (
                self.model,
                self.state_dev_map,
            ),
        )

    @staticmethod
    def _rebuild(model, state_dev_map):
        sdm = state_dev_map
        st = model.state_dict(keep_vars=True)
        st = {k: v.to(sdm[k]) for k, v in st.items()}
        model.load_state_dict(st, strict=False, assign=True)
        return model


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


def _reduce_bnb_param(p):
    state = p.__getstate__()
    return (_rebuild_bnb_param, (type(p), state))


def _rebuild_bnb_param(typ, state):
    v = typ.__new__(typ)
    v.__setstate__(state)
    return v


def bitsandbytes_quirks():
    try:
        import bitsandbytes
    except ImportError as e:
        return

    ForkingPickler.register(bitsandbytes.nn.modules.Params4bit, _reduce_bnb_param)
    ForkingPickler.register(bitsandbytes.nn.modules.Int8Params, _reduce_bnb_param)


def pytorch_pickle_quirks():
    import torch.nn
    forward = torch.nn.ModuleList.forward
    forward.__name__ = 'forward'


def init_reductions_client():
    ForkingPickler.register(memoryview, reduce_memoryview_on_client)
    pytorch_pickle_quirks()
    bitsandbytes_quirks()


def init_reductions_server():
    ForkingPickler.register(memoryview, reduce_memoryview_on_server)
    pytorch_pickle_quirks()
    bitsandbytes_quirks()
