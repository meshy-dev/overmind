# -*- coding: utf-8 -*-

# -- stdlib --
from multiprocessing.connection import Connection, Listener
from multiprocessing.pool import ThreadPool
from pathlib import Path
import sys
import argparse
import importlib
import logging
import multiprocessing.reduction
import os
import sys
import threading
import time
import traceback
import uuid

# -- third party --
# -- own --
from . import reducer
from .common import OvermindEnv, OvermindObjectRef, ServiceExceptionInfo, display_of, key_of


# -- code --
Pickler = multiprocessing.reduction.ForkingPickler
log = logging.getLogger('overmind.server')


class OvermindService:

    def __init__(self):
        super().__init__()
        self._models = {}
        self._models_byref = {}
        self._models_disp = []  # solely for debugging
        self._loading = threading.Lock()
        self._tracked_memoryviews = {}
        self._tracked_wrapped = {}

        reducer.current_service = self

    def exposed_ping(self):
        return 'pong'

    def exposed_load(self, v, args, kwargs):
        if isinstance(v, tuple):
            # This makes pickle happy
            m, n = v
            fn = getattr(importlib.import_module(m), n)
        else:
            fn = v

        key = key_of(fn, args, kwargs)
        disp = display_of(fn, args, kwargs)

        if key in self._models:
            payload = bytes(Pickler.dumps(self._models[key]))
            log.debug('Providing cached model %s (%s bytes over wire)', disp, len(payload))
            return payload

        with self._loading:
            if key in self._models:
                log.debug('Providing cached model (just loaded!) %s', disp)
                return bytes(Pickler.dumps(self._models[key]))

            kwargs = self._convert_refs(kwargs)
            log.info('Cold load model %s', disp)
            b4 = time.time()
            model = fn(*args, **kwargs)

            rid = str(uuid.uuid4())
            try:
                model._overmind_ref = rid
            except AttributeError:
                pass

            model = self._transform(model)
            log.info('Model %s loaded in %.3fs', disp, time.time() - b4)
            self._models[key] = model

            self._models_byref[rid] = model
            self._models_disp.append(disp)
            data = bytes(Pickler.dumps(self._models[key]))
            log.info(f'Will send {len(data)} bytes')
            return data

    def _transform(self, model):
        model = self._set_no_grad(model)
        model = self._move_torch_module_to_cpu(model)
        return model

    def _set_no_grad(self, model):
        import torch
        if not isinstance(model, torch.nn.Module):
            return model

        for p in model.parameters():
            p.requires_grad = False

        return model

    def _move_torch_module_to_cpu(self, model):
        import torch

        if not isinstance(model, torch.nn.Module):
            return model

        dev_map = {}

        def walk(prefix, module):
            for n, v in module.named_parameters(prefix=prefix, recurse=False):
                dev_map[n] = v.device
                if v.device.type == 'cpu':
                    continue
                v1 = v.to('cpu')
                if type(v) is torch.nn.Parameter:
                    v1 = torch.nn.Parameter(v1)
                assert type(v) is type(v1)
                setattr(module, n.rsplit('.')[-1], v1)

            for n, v in module.named_buffers(prefix=prefix, recurse=False):
                dev_map[n] = v.device
                if v.device.type == 'cpu':
                    continue
                setattr(module, n.rsplit('.')[-1], v.to('cpu'))

            for n, v in module.named_children():
                walk(n, v)

        walk('', model)

        # Remove AlignDevices hooks
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, True)

        # Remove accelerate added warning hooks (interferes pickling)
        model.__dict__.pop('to', None)
        model.__dict__.pop('cuda', None)
        model.__dict__.pop('xpu', None)
        model.__dict__.pop('npu', None)

        # Do not wrap pure CPU models
        if set(dev_map.values()) == {torch.device("cpu")}:
            return model

        # Drop VRAM usage at best
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return reducer.RebuildTorchModuleOnClient(model, dev_map)

    def exposed_shutdown(self):
        log.info('!! Bye')
        os._exit(0)

    def exposed_list_loaded(self):
        return self._models_disp

    def exposed_drop_shell(self):
        import IPython
        IPython.embed()

    def _convert_refs(self, obj):
        if isinstance(obj, (list, tuple)):
            return obj.__class__([self._convert_refs(x) for x in obj])
        elif isinstance(obj, dict):
            return {k: self._convert_refs(v) for k, v in obj.items()}
        elif isinstance(obj, OvermindObjectRef):
            return self._models_byref[str(obj)]
        else:
            return obj


class ThreadedServer:

    def __init__(self, service):
        self.service = service
        self.pool = ThreadPool(16)

    def run(self):
        omenv = OvermindEnv.get()
        listener = Listener(omenv.comm_endpoint, authkey=omenv.venv_hash.encode('utf-8'))
        log.info('Overmind server started at %s', omenv.comm_endpoint.replace("\x00", "@"))
        while True:
            client = listener.accept()
            self.pool.apply_async(self._serve, [client])

    def _serve(self, client: Connection):
        try:
            while True:
                fn, args, kwargs = client.recv()
                f = getattr(self.service, f'exposed_{fn}')
                try:
                    ret = f(*args, **kwargs)
                except Exception as e:
                    log.exception(f'Error calling {fn}')
                    text = traceback.format_exc()
                    client.send(ServiceExceptionInfo(type=type(e), desc=str(e), traceback=text))
                    continue

                client.send(ret)
        except (EOFError, OSError):
            pass
        except Exception:
            log.exception('Serve error')


def main():
    import torch.multiprocessing as mp

    mp.set_sharing_strategy('file_system')

    import overmind.reducer
    overmind.reducer.init_reductions_server()

    server = ThreadedServer(OvermindService())
    server.run()


def daemon_main():
    from overmind.utils.log import init as init_log
    omenv = OvermindEnv.get()
    init_log(logging.DEBUG, f'/tmp/overmind.{omenv.venv_hash}.log')
    main()


def start():
    from . import common
    assert common.IN_OVERMIND_SERVER is None, 'Should not import both client and server'
    common.IN_OVERMIND_SERVER = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--daemon', action='store_true')
    parser.add_argument('--fork', action='store_true')
    options = parser.parse_args()

    from overmind.utils.log import init as init_log
    omenv = OvermindEnv.get()

    assert 'torch' not in sys.modules

    if options.daemon:
        assert sys.platform == 'linux'
        from daemonize import Daemonize
        pid = Path(f'/tmp/overmind.{omenv.venv_hash}.pid')
        if pid.exists():
            pid.unlink()
        daemon = Daemonize(app="overmind", pid=str(pid), action=daemon_main, logger=logging.getLogger('daemonize'))
        daemon.start()
    elif options.fork:
        if os.fork():
            return
        os.setsid()
        init_log(logging.DEBUG, f'/tmp/overmind.{omenv.venv_hash}.log')
        main()
    else:
        init_log(logging.DEBUG, None)
        main()


if __name__ == '__main__':
    start()
