# -*- coding: utf-8 -*-

# -- stdlib --
from functools import lru_cache
from multiprocessing.connection import Client
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
# -- own --
from . import common
from .common import OvermindEnv, ServiceCaller, display_of, key_of
from .utils.misc import hook, walk_obj


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
            return ServiceCaller(self.client).call(fn, *args, **kwargs)

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
            e = OvermindEnv.get()
            self.client = Client(e.comm_endpoint, authkey=e.venv_hash.encode('utf-8'))
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

    def _local_cached_load(self, fn, args, kwargs):
        if os.environ.get('OVERMIND_NO_LOCAL_CACHE'):
            return fn(*args, **kwargs)

        key = key_of(fn, args, kwargs)
        if key in self._local_cache:
            return self._local_cache[key]
        self._local_cache[key] = ret = fn(*args, **kwargs)
        return ret

    def load(self, fn, *args, **kwargs):
        from .reducer import OvermindUnpickler as Unpickler
        from .reducer import ForkingPickler as Pickler
        from .shmem import borrower

        if os.environ.get('OVERMIND_DISABLE'):
            log.warning('overmind disabled by OVERMIND_DISABLE env variable, loading model directly')
            return self._local_cached_load(fn, args, kwargs)

        if not self.enabled:
            return self._local_cached_load(fn, args, kwargs)

        self._init_client()

        def replace_ref(obj):
            if (ref := getattr(obj, '_overmind_ref', None)):
                return False, ref
            return True, obj

        fn, args, kwargs = walk_obj((fn, args, kwargs), pre=replace_ref)
        key = key_of(fn, args, kwargs)
        disp = display_of(fn, args, kwargs)

        if isinstance(fn, types.FunctionType):
            fn = (fn.__module__, fn.__qualname__)

        b4 = time.time()
        arenas, b = self._call('load', bytes(Pickler.dumps(fn)), args, kwargs, key, disp)
        rpc_time = time.time() - b4
        borrower.import_arenas(arenas)
        obj = Unpickler.loads(Unpickler.loads(b))
        log.info(f'Loaded {disp} in {time.time() - b4:.3f}s (rpc: {rpc_time:.3f}s)')
        return obj


om = OvermindClient()
load = om.load


@lru_cache(None)
def monkey_patch(spec):
    if common.IN_OVERMIND_SERVER:
        return

    if os.environ.get('OVERMIND_DISABLE'):
        return

    try:
        modulename, attrname = spec.split('::')
        module = importlib.import_module(modulename)

        if '.' in attrname:
            clsname, method = attrname.split('.')
            target = getattr(module, clsname)
            name = method
        else:
            target = module
            name = attrname
    except (ModuleNotFoundError, AttributeError):
        log.warn(f'Could not find {spec}, monkey patching skipped')
        return

    hook(target, name=name)(load)
    log.info(f'Patched {spec}')


@lru_cache(None)
def monkey_patch_from_config_file(path):
    path = Path(path)

    if not path.exists():
        log.warning(f'Config file {path} does not exist, skipping monkey patching')
        return

    log.info(f':: Patching from config file {path}')

    lines = path.read_text().splitlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line and not line.startswith('#')]

    for spec in lines:
        monkey_patch(spec)



@lru_cache(1)
def monkey_patch_all():
    if common.IN_OVERMIND_SERVER:
        return

    if os.environ.get('OVERMIND_DISABLE'):
        log.warning('overmind disabled by OVERMIND_DISABLE env variable, not monkey patching')
        return

    from .assets import assets

    monkey_patch_from_config_file(assets / 'predefined.cfg')

    caller = sys._getframe(1).f_code.co_filename
    try:
        caller = Path(caller)
        if not caller.exists():
            return
        path = caller.resolve().parent
    except Exception:
        return

    while (path / '__init__.py').exists():
        if (cfg := path / 'overmind.cfg').exists():
            monkey_patch_from_config_file(cfg)
        path = path.parent


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
    if common.IN_OVERMIND_SERVER is True:
        return
    common.IN_OVERMIND_SERVER = False
    apply_quirks()
    from .reducer import init_reductions_client
    init_reductions_client()

_init()
