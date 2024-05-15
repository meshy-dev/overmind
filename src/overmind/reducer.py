# -*- coding: utf-8 -*-

# -- stdlib --
from _thread import _local
from multiprocessing.reduction import ForkingPickler
import io
import zipfile

# -- third party --
import dill

# -- own --
from .shmem import Fragment
from .utils.misc import hook


# -- code --
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


def _rebuild_memoryview_on_client(v: Fragment):
    from .shmem import borrower
    return borrower.borrow(v)


def _reduce_memoryview_on_client(v: memoryview):
    return (memoryview, (bytes(v),))


def _reduce_memoryview_on_server(v: memoryview):
    from .shmem import hoarder
    frag = hoarder.put(v)
    return (_rebuild_memoryview_on_client, (frag,))


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


def _rebuild_torch_jit_objects(payload: memoryview):
    from torch.jit._recursive import wrap_cpp_module
    import torch._C
    import overmind._C

    cu = torch._C.CompilationUnit()
    cpp_module = overmind._C.import_ir_module_from_buffer_0copy(cu, payload)
    return wrap_cpp_module(cpp_module)


def _reduce_torch_jit_objects(obj):
    zipped = io.BytesIO()
    obj.save(zipped)

    zipped.seek(0)
    inflated = io.BytesIO()

    # Inflate the zip file to speed up loading
    with zipfile.ZipFile(inflated, 'w', zipfile.ZIP_STORED) as o:
        with zipfile.ZipFile(zipped, 'r') as i:
            for f in i.infolist():
                o.writestr(f, i.read(f.filename))

    return (_rebuild_torch_jit_objects, (memoryview(inflated.getvalue()),))


def _rebuild_storage_on_client(frag, device):
    from .shmem import borrower
    from overmind._C import _make_untyped_storage
    mv = borrower.borrow(frag)
    storage = _make_untyped_storage(mv)
    if device == 'cpu':
        return storage
    elif str(device).startswith('cuda'):
        return storage.cuda(device)


def _reduce_storage(storage):
    # Copied from torch.multiprocessing.reductions, with modifications

    from torch.multiprocessing.reductions import rebuild_storage_empty
    from .shmem import hoarder

    if storage.size() == 0:
        # This is special cased because Empty tensors
        # (with size 0) cannot be mmapped.
        return (rebuild_storage_empty, (type(storage),))
    else:
        frag = hoarder.put(bytes(storage))
        return (_rebuild_storage_on_client, (frag, storage.device))


def _reduce_tensor(tensor):
    # Copied from torch.multiprocessing.reductions, with modifications
    # - CUDA sharing is removed
    # - Requires requires_grad == False

    from torch.multiprocessing.reductions import check_serializing_named_tensor, rebuild_tensor
    import torch.utils.hooks

    storage = tensor._typed_storage()

    if tensor.requires_grad:
        raise RuntimeError(
            "Tensors with requires_grad=True does not make sense in overmind, please fix it"
        )

    check_serializing_named_tensor(tensor)
    torch.utils.hooks.warn_if_has_hooks(tensor)

    # _backward_hooks purposely omitted here, see Note [Don't serialize hooks]
    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        tensor.requires_grad,
    )
    return (rebuild_tensor, (type(tensor), storage, metadata))


def pytorch_pickle_quirks(*, server: bool):
    import torch.nn
    forward = torch.nn.ModuleList.forward
    forward.__name__ = 'forward'

    import torch.jit
    import torch.multiprocessing.reductions

    if server:
        ForkingPickler.register(torch.jit.RecursiveScriptModule, _reduce_torch_jit_objects)  # noqa
        # ForkingPickler.register(torch.jit.RecursiveScriptClass, _reduce_torch_jit_objects)
        # ForkingPickler.register(torch.jit.ScriptObject, _reduce_torch_jit_objects)
        ForkingPickler.register(torch.jit.ScriptModule, _reduce_torch_jit_objects)
        ForkingPickler.register(torch.jit.ScriptFunction, _reduce_torch_jit_objects)

        ForkingPickler.register(torch.Tensor, _reduce_tensor)
        ForkingPickler.register(torch.UntypedStorage, _reduce_storage)


def stable_fast_quirks():
    try:
        import sfast.jit.utils
        import sfast.triton.torch_ops  # noqa
        import sfast.utils.flat_tensors
    except ImportError:
        return

    sfast.jit.utils.attach_script_module_clear_hook = lambda *_, **__: None

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
    ForkingPickler.register(memoryview, _reduce_memoryview_on_client)
    thread_quirks()
    pytorch_pickle_quirks(server=False)
    stable_fast_quirks()
    bitsandbytes_quirks()


def init_reductions_server():
    ForkingPickler.register(memoryview, _reduce_memoryview_on_server)
    thread_quirks()
    pytorch_pickle_quirks(server=True)
    stable_fast_quirks()
    bitsandbytes_quirks()
