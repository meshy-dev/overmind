# -*- coding: utf-8 -*-

# -- stdlib --
from multiprocessing.reduction import ForkingPickler
import io

# -- third party --
import dill

# -- own --
from .shmem import SharedMemory
from .utils.misc import hook


# -- code --
current_service = None


class OvermindPickler(dill.Pickler):

    def __init__(self, file):
        super().__init__(file)
        self.dispatch_table = ForkingPickler(file).dispatch_table

    @classmethod
    def dumps(cls, obj):
        buf = io.BytesIO()
        cls(buf).dump(obj)
        return buf.getbuffer()


class RebuildTorchModuleOnClient:

    def __init__(self, model, dev_map):
        self.model = model
        self.dev_map = dev_map

    def __reduce__(self):
        return (
            self._rebuild,
            (self.model, self.dev_map),
        )

    @staticmethod
    def _rebuild(model, dev_map):
        import torch

        def walk(prefix, module):
            for n, v in module.named_parameters(prefix=prefix, recurse=False):
                v1 = v.to(dev_map[n])
                if type(v) is torch.nn.Parameter:
                    v1 = torch.nn.Parameter(v1)

                assert type(v) is type(v1)
                setattr(module, n.rsplit('.')[-1], v1)

            for n, v in module.named_buffers(prefix=prefix, recurse=False):
                setattr(module, n.rsplit('.')[-1], v.to(dev_map[n]))

            for n, v in module.named_children():
                walk(n, v)

        walk('', model)
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
    assert p.quant_state
    qs_dict = p.quant_state.as_dict(packed=True)
    return (_rebuild_bnb_param, (type(p), p.data, qs_dict))


def _rebuild_bnb_param(typ, data, qs_dict):
    return typ.from_prequantized(data, qs_dict, device='cpu')


def bitsandbytes_quirks():
    try:
        import bitsandbytes
    except ImportError as e:
        return

    ForkingPickler.register(bitsandbytes.nn.modules.Params4bit, _reduce_bnb_param)
    ForkingPickler.register(bitsandbytes.nn.modules.Int8Params, _reduce_bnb_param)

    @hook(bitsandbytes.nn.modules.QuantState)
    def to(orig, self, device):
        orig(self, device)
        self.code = self.code.to(device)


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
