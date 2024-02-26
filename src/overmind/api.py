# -*- coding: utf-8 -*-

# -- stdlib --
from functools import lru_cache
from multiprocessing.reduction import ForkingPickler as Pickler
from pathlib import Path
from typing import Any
import fcntl
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import time

# -- third party --
import rpyc.core.protocol
import rpyc.utils.factory
import torch.multiprocessing as mp

# -- own --
from .common import OvermindObjectRef
from .utils.misc import hook


# -- code --
log = logging.getLogger('overmind.api')
rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout'] = 900


class OvermindClient:

    def __init__(self):
        self.client: Any = None
        self.enabled = True

    def _is_client_ok(self):
        if not self.client:
            return False

        try:
            return self.client.root.ping() == 'pong'
        except Exception:
            return False

    def _try_connect(self):
        try:
            log.debug('Try connecting to overmind server...')
            self.client = rpyc.utils.factory.unix_connect('/tmp/overmind.sock')
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

        if sys.platform == 'win32':
            log.warning('Overmind server will not auto start on Windows, please start it manually. Falling back to local mode.')
            self.enabled = False
            return

        with open('/tmp/overmind.lock', 'w') as lockf:
            while True:
                try:
                    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.3)
                    self._try_connect()
                    if self._is_client_ok():
                        return

            self._try_connect()
            if self._is_client_ok():
                return

            log.debug(f'[pid {os.getpid()}] Starting overmind server as daemon...')
            # if os.system(f'{sys.executable} -m overmind.server --daemon') != 0:
            if os.system('overmind-server --fork') != 0:
                raise RuntimeError('Failed to start overmind server')

            time.sleep(0.5)

            try:
                log.debug('Connecting to newly spawned overmind server...')
                self.client = rpyc.utils.factory.unix_connect('/tmp/overmind.sock')
            except Exception:
                log.exception('Failed to spawn overmind server')

            if self._is_client_ok():
                return

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

    def load(self, fn, *args, **kwargs):
        self._init_client()

        if not self.enabled:
            return fn(*args, **kwargs)

        s = inspect.signature(fn)
        bs = s.bind(*args, **kwargs)
        kwargs = bs.arguments

        for k in kwargs:
            if s.parameters[k].kind == inspect.Parameter.VAR_KEYWORD:
                kwargs.update(kwargs.pop(k))
                break

        payload = self._convert_to_refs((fn, kwargs))

        b: bytes = self.client.root.load(bytes(Pickler.dumps(payload)))
        return Pickler.loads(b)


om = OvermindClient()
load = om.load


def monkey_patch(modulename, clsname, method):
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
    import torch

    def hook_load(orig, f, map_location=None, **kwargs):
        if map_location == 'cpu':
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
    monkey_patch('diffusers.pipelines.pipeline_utils',   'DiffusionPipeline',       'from_pretrained')
    monkey_patch('diffusers.models.modeling_utils',      'ModelMixin',              'from_pretrained')
    monkey_patch('diffusers.schedulers.scheduler_utils', 'SchedulerMixin',          'from_pretrained')
    monkey_patch('transformers.modeling_utils',          'PreTrainedModel',         'from_pretrained')
    monkey_patch('transformers.tokenization_utils_base', 'PreTrainedTokenizerBase', 'from_pretrained')
    monkey_patch('transformers',                         'AutoProcessor',           'from_pretrained')
    # monkey_patch('torchvision.models.vgg',               None,                      'vgg19')
    # monkey_patch('torchvision.models.vgg',               None,                      'vgg16')
    # monkey_patch('open_clip',                            None,                      'create_model_and_transforms')

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


_init()
