# -*- coding: utf-8 -*-

# -- stdlib --
from _thread import _local
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing.reduction import ForkingPickler
import io
import logging
import zipfile

# -- third party --
import dill

# -- own --
from .shmem import Fragment


# -- code --
log = logging.getLogger(__name__)


@dataclass
class OvermindRef:
    key: str
    disp: str

    def __repr__(self):
        return f'<{self.key} &{self.disp}>'


class OvermindPickler(dill.Pickler):
    _my_extra_reducers = {}

    def __init__(self, file):
        super().__init__(file)
        self.dispatch_table = ForkingPickler(file).dispatch_table
        self.dispatch_table.update(self._my_extra_reducers)

    @classmethod
    def dumps(cls, obj):
        buf = io.BytesIO()
        cls(buf).dump(obj)
        return buf.getbuffer()

    @classmethod
    def register(cls, type, reduce):
        cls._my_extra_reducers[type] = reduce


class OvermindUnpickler(ForkingPickler):
    def __init__(self, *args):
        super().__init__(*args)
        self.dispatch_table.update(OvermindPickler._my_extra_reducers)


def _rebuild_memoryview_on_client(v: Fragment):
    from .shmem import borrower
    return borrower.borrow(v)


def _reduce_memoryview_on_client(v: memoryview):
    return (memoryview, (bytes(v),))


def _reduce_memoryview_on_server(v: memoryview):
    from .shmem import hoarder
    frag = hoarder.put(v)
    return (_rebuild_memoryview_on_client, (frag,))


def _rebuild_torch_jit_objects(payload: memoryview):
    from torch.jit._recursive import wrap_cpp_module
    import torch._C
    import overmind._C

    cu = torch._C.CompilationUnit()
    cpp_module = overmind._C.import_ir_module_from_buffer_0copy(cu, payload)
    return wrap_cpp_module(cpp_module)


def _reduce_torch_jit_objects(obj):
    import torch
    zipped = io.BytesIO()
    torch.jit.save(obj, zipped)
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
    if device.type == 'cpu':
        return storage
    elif device.type == 'cuda':
        return storage.cuda(device.index)
    else:
        raise ValueError(f'Unexpected device {repr(device)}')


def _reduce_storage(storage):
    # Copied from torch.multiprocessing.reductions, with modifications

    from torch.multiprocessing.reductions import rebuild_storage_empty
    from .shmem import hoarder

    if storage.size() == 0:
        # This is special cased because Empty tensors
        # (with size 0) cannot be mmapped.
        return (rebuild_storage_empty, (type(storage),))
    else:
        device = storage.device
        storage = storage.cpu()
        frag = hoarder.put(storage)
        return (_rebuild_storage_on_client, (frag, device))


@lru_cache(1)
def _warn_requires_grad():
    log.warning(
        "Tensors with requires_grad=True does not make sense in overmind, will be set to False automatically. "
        "Subsequent warnings will be suppressed."
    )


def _reduce_tensor(tensor):
    # Copied from torch.multiprocessing.reductions, with modifications
    # - CUDA sharing is removed
    # - Sets requires_grad == False

    from torch.multiprocessing.reductions import check_serializing_named_tensor, rebuild_tensor
    import torch.utils.hooks

    storage = tensor._typed_storage()

    if tensor.requires_grad:
        _warn_requires_grad()

    check_serializing_named_tensor(tensor)
    torch.utils.hooks.warn_if_has_hooks(tensor)

    # _backward_hooks purposely omitted here, see Note [Don't serialize hooks]
    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        False and tensor.requires_grad,
    )
    return (rebuild_tensor, (type(tensor), storage, metadata))


def pytorch_pickle_quirks(*, server: bool):
    import torch.nn
    forward = torch.nn.ModuleList.forward
    forward.__name__ = 'forward'

    import torch.jit
    import torch.multiprocessing.reductions

    def register_server(cls, fn):
        OvermindPickler.register(cls, fn)

    def forbid_reducing(obj):
        raise RuntimeError(
            f"You should not pickle a {type(obj)} object "
            "(or sending it to overmind, which is basically the same thing). "
            "This is a limitation of overmind."
        )

    def register_client(cls, _fn):
        OvermindPickler.register(cls, forbid_reducing)

    if server:
        register = register_server
    else:
        register = register_client

    register(torch.jit.RecursiveScriptModule, _reduce_torch_jit_objects)  # noqa
    # register(torch.jit.RecursiveScriptClass, _reduce_torch_jit_objects)
    # register(torch.jit.ScriptObject, _reduce_torch_jit_objects)
    register(torch.jit.ScriptModule, _reduce_torch_jit_objects)
    register(torch.jit.ScriptFunction, _reduce_torch_jit_objects)

    for t in torch._tensor_classes:
        register(t, _reduce_tensor)
    register(torch.Tensor, _reduce_tensor)
    register(torch.nn.parameter.Parameter, _reduce_tensor)

    register(torch.UntypedStorage, _reduce_storage)


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
    OvermindPickler.register(_local, lambda _: (_local, ()))


def init_reductions_client():
    ForkingPickler.register(memoryview, _reduce_memoryview_on_client)  # Not OvermindPickler, it's not a typo
    thread_quirks()
    pytorch_pickle_quirks(server=False)
    stable_fast_quirks()


def init_reductions_server():
    ForkingPickler.register(memoryview, _reduce_memoryview_on_server)  # Not OvermindPickler, it's not a typo
    thread_quirks()
    pytorch_pickle_quirks(server=True)
    stable_fast_quirks()
