# -*- coding: utf-8 -*-

# -- stdlib --
from multiprocessing.reduction import ForkingPickler
from _thread import _local
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
    except ImportError:
        return

    ForkingPickler.register(bitsandbytes.nn.modules.Params4bit, _reduce_bnb_param)
    ForkingPickler.register(bitsandbytes.nn.modules.Int8Params, _reduce_bnb_param)

    @hook(bitsandbytes.nn.modules.QuantState)
    def to(orig, self, device):
        orig(self, device)
        self.code = self.code.to(device)


class RebuildTorchJitOnClient:

    def __init__(self, mod):
        self.serialized = self.reduce_torch_jit_objects(mod)

    def __reduce__(self):
        return self.serialized

    @staticmethod
    def rebuild_torch_jit_objects(data):
        ...  # Prevent string below to be recognized as docstring

        # Flatbuffer version does not preserve device, not using it for now
        '''
        import torch._C
        from torch.jit._recursive import wrap_cpp_module
        # FIXME: data here is copied 2 times, each time cost 0.5s for a UNet model
        # 1st copy happens here, since _load_jit_module_from_bytes will not accept memoryview objects
        # 2nd copy happens in C++ code of _load_jit_module_from_bytes
        data = bytes(data)
        rst = wrap_cpp_module(torch._C._load_jit_module_from_bytes(data))
        '''

        import torch
        rst = torch.jit.load(io.BytesIO(data))
        return rst

    @staticmethod
    def reduce_torch_jit_objects(obj, rebuild_torch_jit_objects=rebuild_torch_jit_objects):
        import torch.jit
        b = io.BytesIO()

        import objexplore

        def exp(obj):
            # Debug code
            import types
            while (v := objexplore.explore(obj)) is not None:
                try:
                    if isinstance(v, (types.FunctionType, types.MethodType)):
                        v = v()

                    if isinstance(v, str):
                        print(v)
                        print('-------------------------')
                        input()
                    elif isinstance(v, types.GeneratorType):
                        exp(list(v))
                    else:
                        exp(v)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    print('-------------------------')
                    input()

        # exp(obj)
        # import IPython
        # IPython.embed()
        # data = memoryview(torch._C._save_jit_module_to_bytes(obj._c, {}))
        data = memoryview(obj.save_to_buffer())
        return (rebuild_torch_jit_objects, (data,))


def pytorch_pickle_quirks():
    import torch.nn
    forward = torch.nn.ModuleList.forward
    forward.__name__ = 'forward'

    import torch.jit

    # import IPython
    # ForkingPickler.register(torch.jit.RecursiveScriptModule, lambda v: IPython.embed())

    # FIXME: Handled at post process for now, revert it when non-shm tensor pickling is implemented
    # ForkingPickler.register(torch.jit.RecursiveScriptModule, reduce_torch_jit_objects)
    # # ForkingPickler.register(torch.jit.RecursiveScriptClass, reduce_torch_jit_objects)
    # # ForkingPickler.register(torch.jit.ScriptObject, reduce_torch_jit_objects)
    # ForkingPickler.register(torch.jit.ScriptModule, reduce_torch_jit_objects)
    # ForkingPickler.register(torch.jit.ScriptFunction, reduce_torch_jit_objects)


def stable_fast_quirks():
    try:
        import sfast.jit.utils
        import sfast.triton.torch_ops  # noqa
        import sfast.utils.flat_tensors
    except ImportError:
        return

    sfast.jit.utils.attach_script_module_clear_hook =lambda *_, **__: None

    # pickle dataclass type instead of just put it into a container (which will not survive after torch.jit.save)
    def flatten_dataclass(obj):
        from sfast.utils.flat_tensors import flatten_bytes, flatten_dict
        import dataclasses
        d = dict((field.name, getattr(obj, field.name))
                for field in dataclasses.fields(obj))
        import pickle
        pickled = pickle.dumps(obj.__class__)
        return flatten_bytes(pickled) + flatten_dict(d)

    def unflatten_dataclass(tensors, start):
        from sfast.utils.flat_tensors import unflatten_bytes, unflatten_dict
        import pickle
        pickled, start = unflatten_bytes(tensors, start)
        clz = pickle.loads(pickled)
        content, start = unflatten_dict(tensors, start)
        return clz(**content), start

    sfast.utils.flat_tensors.flatten_dataclass = flatten_dataclass
    sfast.utils.flat_tensors.unflatten_dataclass = unflatten_dataclass


def thread_quirks():
    # Assuming data in thread local is not important, just drop them
    ForkingPickler.register(_local, lambda _: (_local, ()))


def init_reductions_client():
    ForkingPickler.register(memoryview, reduce_memoryview_on_client)
    thread_quirks()
    pytorch_pickle_quirks()
    stable_fast_quirks()
    bitsandbytes_quirks()


def init_reductions_server():
    ForkingPickler.register(memoryview, reduce_memoryview_on_server)
    thread_quirks()
    pytorch_pickle_quirks()
    stable_fast_quirks()
    bitsandbytes_quirks()
