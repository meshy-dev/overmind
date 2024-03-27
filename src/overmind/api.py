# -*- coding: utf-8 -*-

# -- stdlib --
from functools import lru_cache
from multiprocessing.connection import Client
from multiprocessing.reduction import ForkingPickler as Pickler
from pathlib import Path
from typing import Any
import importlib
import importlib.util
import logging
import os
import sys
import threading
import time
import types

# -- third party --
import torch.multiprocessing as mp

# -- own --
from . import common
from .common import OvermindEnv, OvermindObjectRef, ServiceExceptionInfo, key_of
from .utils.misc import hook


# -- code --
log = logging.getLogger('overmind.api')


class OvermindClient:

    def __init__(self):
        self.client: Any = None
        self.enabled = True
        self._local_cache = {}
        self._client_lock = threading.Lock()

    def _call(self, fn, *args, **kwargs):
        if not self.client:
            raise Exception('Not connected')

        with self._client_lock:
            self.client.send((fn, args, kwargs))
            ret = self.client.recv()
            if isinstance(ret, ServiceExceptionInfo):
                raise ret.to_exception()

        return ret

    def _is_client_ok(self):
        if not self.client:
            return False

        try:
            return self._call('ping') == 'pong'
        except Exception:
            return False

    def _try_connect(self):
        try:
            log.debug('Try connecting to overmind server...')
            self.client = Client(OvermindEnv.get().comm_endpoint)
        except Exception:
            pass

    def _init_client(self):
        if not self.enabled:
            return

        if self._is_client_ok():
            return

        self._try_connect()

        if self._is_client_ok():
            return

        omenv = OvermindEnv.get()

        if sys.platform == 'win32':
            from .utils.win32mutex import Win32Mutex
            mutex = Win32Mutex(omenv.lock_path)
            while True:
                if mutex.acquire():
                    break
                time.sleep(0.3)
                self._try_connect()
                if self._is_client_ok():
                    mutex.release()
                    return
        else:
            lockf = open(omenv.lock_path, 'w')
            while True:
                try:
                    import fcntl

                    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.3)
                    self._try_connect()
                    if self._is_client_ok():
                        lockf.close()
                        return

        try:
            self._try_connect()
            if self._is_client_ok():
                return

            # if os.system(f'{sys.executable} -m overmind.server --daemon') != 0:
            if sys.platform == 'win32':
                log.debug(f'[pid {os.getpid()}] Starting overmind server...')
                os.startfile('overmind-server')
            else:
                mode = ('daemon', 'fork')[os.isatty(1)]
                log.debug(f'[pid {os.getpid()}] Starting overmind server as {mode}...')
                if os.system(f'overmind-server --{mode}') != 0:
                    raise RuntimeError('Failed to start overmind server')

            for _ in range(5):
                time.sleep(1)
                self._try_connect()
                if self._is_client_ok():
                    return
            else:
                log.error('Failed to spawn overmind server')
        finally:
            if sys.platform == 'win32':
                mutex.release()
            else:
                lockf.close()

        log.warning('Could not connect to overmind server, falling back to local mode')
        self.enabled = False

    def _convert_to_refs(self, obj):
        if isinstance(obj, (list, tuple)):
            return obj.__class__([self._convert_to_refs(x) for x in obj])
        elif isinstance(obj, dict):
            return {k: self._convert_to_refs(v) for k, v in obj.items()}
        elif (rid := getattr(obj, '_overmind_ref', None)) is not None:
            return OvermindObjectRef(str(rid))
        else:
            return obj

    def _local_cached_load(self, fn, args, kwargs):
        if os.environ.get('OVERMIND_NO_LOCAL_CACHE'):
            return fn(*args, **kwargs)

        key = key_of(fn, args, kwargs)
        if key in self._local_cache:
            return self._local_cache[key]
        self._local_cache[key] = ret = fn(*args, **kwargs)
        return ret

    def load(self, fn, *args, **kwargs):
        if os.environ.get('OVERMIND_DISABLE'):
            log.warning('overmind disabled by OVERMIND_DISABLE env variable, loading model directly')
            return self._local_cached_load(fn, args, kwargs)

        # Heuristics
        if kwargs.get('load_in_4bit') or kwargs.get('load_in_8bit'):
            log.warning('Does not support load_in_[48]bit for now, loading model directly')
            return self._local_cached_load(fn, args, kwargs)
        # End of heuristcs

        if not self.enabled:
            return self._local_cached_load(fn, args, kwargs)

        self._init_client()

        if isinstance(fn, types.FunctionType):
            # This makes pickle happy
            fn = (fn.__module__, fn.__name__)

        fn, args, kwargs = self._convert_to_refs((fn, args, kwargs))

        b: bytes = self._call('load', fn, args, kwargs)
        return Pickler.loads(b)


om = OvermindClient()
load = om.load


def monkey_patch(modulename, clsname, method):
    if common.IN_OVERMIND_SERVER:
        return

    if os.environ.get('OVERMIND_DISABLE'):
        return

    try:
        module = importlib.import_module(modulename)
        if clsname is None:
            cls = module
        else:
            cls = getattr(module, clsname)
    except (ModuleNotFoundError, AttributeError):
        log.info(f'Could not import {modulename}.{clsname}, skipping monkey patching')
        return

    hook(cls, name=method)(load)
    log.info(f'Patched {modulename}.{clsname}.{method}')


def monkey_patch_torch_load():
    if common.IN_OVERMIND_SERVER:
        return

    import torch

    def hook_load(orig, f, map_location=None, **kwargs):
        if map_location in ('cpu', torch.device("cpu")):
            return load(orig, f, map_location, **kwargs)
        elif map_location is None:
            log.warning('torch.load called with map_location=None, aggressively assuming to load on CPU')
            return load(orig, f, 'cpu', **kwargs)
        else:
            log.warning('torch.load called with map_location != "cpu", falling back to local mode')
            return orig(f, map_location, **kwargs)

    hook(torch, name='load')(hook_load)
    hook(torch.jit, name='load')(hook_load)


@lru_cache(1)
def monkey_patch_all():
    if common.IN_OVERMIND_SERVER:
        return

    if os.environ.get('OVERMIND_DISABLE'):
        log.warning('overmind disabled by OVERMIND_DISABLE env variable, not monkey patching')
        return

    monkey_patch('diffusers.pipelines.pipeline_utils',   'DiffusionPipeline',       'from_pretrained')
    monkey_patch('diffusers.models.modeling_utils',      'ModelMixin',              'from_pretrained')
    monkey_patch('diffusers.schedulers.scheduler_utils', 'SchedulerMixin',          'from_pretrained')
    monkey_patch('transformers.modeling_utils',          'PreTrainedModel',         'from_pretrained')
    monkey_patch('transformers.tokenization_utils_base', 'PreTrainedTokenizerBase', 'from_pretrained')
    monkey_patch('transformers',                         'AutoProcessor',           'from_pretrained')
    monkey_patch('torchvision.models.vgg',               None,                      'vgg19')
    monkey_patch('torchvision.models.vgg',               None,                      'vgg16')
    monkey_patch('open_clip',                            None,                      'create_model_and_transforms')
    monkey_patch('safetensors.torch',                    None,                      'load_file')

    monkey_patch_torch_load()


def diffusers_dyn_module_workaround():
    try:
        from diffusers.utils.constants import HF_MODULES_CACHE
    except ImportError:
        return

    modpath = Path(HF_MODULES_CACHE) / "diffusers_modules/__init__.py"

    if not modpath.exists():
        modpath.parent.mkdir(parents=True, exist_ok=True)
        modpath.touch()

    spec = importlib.util.spec_from_file_location("diffusers_modules", modpath)
    assert spec
    foo = importlib.util.module_from_spec(spec)
    sys.modules["diffusers_modules"] = foo


def apply_quirks():
    diffusers_dyn_module_workaround()


def _init():
    mp.set_sharing_strategy('file_system')
    apply_quirks()
    from .reducer import init_reductions_client
    init_reductions_client()


_init()
