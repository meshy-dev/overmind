# -*- coding: utf-8 -*-

# -- stdlib --
from multiprocessing.connection import Connection, Listener
from multiprocessing.pool import ThreadPool
from pathlib import Path
import argparse
import gc
import importlib
import logging
import os
import sys
import threading
import time
import traceback

# -- third party --
# -- own --
from .common import OvermindEnv, ServiceExceptionInfo, display_of, key_of
from .reducer import OvermindPickler


# -- code --
log = logging.getLogger('overmind.server')


class OvermindService:

    def __init__(self):
        super().__init__()
        self._models = {}
        self._models_disp = []  # solely for debugging
        self._loading = threading.Lock()

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

        # Heuristics:
        import torch

        while (dev := kwargs.get('device')):
            if dev in ('cpu', 'cuda', torch.device('cpu')):
                break

            if isinstance(dev, torch.device) and dev.type == 'cuda':
                if dev.index not in (None, 0):
                    raise ValueError(f'Only models on cuda:0 are supported, not loading {disp}')

                kwargs['device'] = torch.device('cuda:0')

            break

        while (dmap := kwargs.get('device_map')):
            if dmap == 'auto':
                if kwargs.get('load_in_4bit') or kwargs.get('load_in_8bit'):
                    log.warning('Auto device_map is not supported, forcing cuda:0 (since 4bit/8bit quant is used)')
                    kwargs['device_map'] = {'': torch.device('cuda:0')}
                else:
                    log.warning('Auto device_map is not supported, forcing cpu')
                    kwargs.pop('device_map')
                    kwargs['device_map'] = {'': torch.device('cpu')}

                break

            elif isinstance(dmap, dict) and len(dmap) == 1:
                dev = next(iter(dmap.values()))

                if dev in ('cpu', 'cuda', torch.device('cpu')):
                    break

                if isinstance(dev, torch.device) and dev.type == 'cuda':
                    if dev.index not in (None, 0):
                        raise ValueError(f'Only models on cuda:0 are supported, not loading {disp}')

                    kwargs['device_map'] = {'': torch.device('cuda:0')}
            else:
                raise ValueError('Complex device_map is not supported')

            break
        # End of heuristics

        if key in self._models:
            payload = self._models[key]
            log.debug('Providing cached model %s (%s bytes over wire)', disp, len(payload))
            return payload

        with self._loading:
            if key in self._models:
                log.debug('Providing cached model (just loaded!) %s', disp)
                return self._models[key]

            log.info('Cold load model %s', disp)
            b4 = time.time()
            model = fn(*args, **kwargs)

            model = self._transform(model)
            log.info('Model %s loaded in %.3fs', disp, time.time() - b4)

            b4 = time.time()
            self._models[key] = OvermindPickler.dumps(OvermindPickler.dumps(model))  # Double pickle, saving pickle to shared mem too!
            log.info('Pickled in %.3fs, size = %s bytes', time.time() - b4, len(self._models[key]))

            self._models_disp.append(disp)

            del model

            gc.collect()
            torch.cuda.empty_cache()

            return self._models[key]

    def _transform(self, model):
        model = self._set_no_grad(model)
        model = self._walk_generic_thing(model)
        return model

    def _set_no_grad(self, model):
        import torch
        if not isinstance(model, torch.nn.Module):
            return model

        for p in model.parameters():
            p.requires_grad = False

        return model

    def _walk_generic_thing(self, obj):
        import torch

        seen = set()
        ref = []

        def walk(m):
            if id(m) in seen:
                return m

            ref.append(m)
            seen.add(id(m))

            if isinstance(m, torch.nn.Module):
                m = self._remove_accelerate_hooks(m)
            elif isinstance(m, list):
                for i, v in enumerate(m):
                    m[i] = walk(v)
            elif isinstance(m, tuple):
                m = m.__class__(walk(v) for v in m)
            elif isinstance(m, dict):
                for k, v in m.items():
                    m[k] = walk(v)
            elif (d := getattr(m, '__dict__', None)) is not None:
                for k, v in d.items():
                    d[k] = walk(v)
            elif (keys := getattr(m, '__slots__', None)) is not None:
                for k in keys:
                    setattr(m, k, walk(getattr(m, k)))

            ref.append(m)
            seen.add(id(m))  # not a duplicate, m may has changed

            return m

        return walk(obj)

    def _remove_accelerate_hooks(self, model):
        import torch

        if not isinstance(model, torch.nn.Module):
            return model

        # Remove AlignDevices hooks
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, True)

        # Remove accelerate added warning hooks (interferes pickling)
        model.__dict__.pop('to', None)
        model.__dict__.pop('cuda', None)
        model.__dict__.pop('xpu', None)
        model.__dict__.pop('npu', None)

        return model

    def exposed_shutdown(self):
        log.info('!! Bye')
        os._exit(0)

    def exposed_list_loaded(self):
        return self._models_disp

    def exposed_drop_shell(self):
        import IPython
        IPython.embed()


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
