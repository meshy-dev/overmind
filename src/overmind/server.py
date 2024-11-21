# -*- coding: utf-8 -*-

# -- stdlib --
from multiprocessing.connection import Connection, Listener
from multiprocessing.pool import ThreadPool
import os
from typing import Any
from pathlib import Path
import multiprocessing
import argparse
import base64
import importlib
import logging
import os
from typing import List
import random
import sys
import threading
import time
import traceback

# -- third party --
# -- own --
from .common import OvermindEnv, ServiceExceptionInfo, display_of, key_of
from .reducer import OvermindPickler, OvermindRef
from .utils.misc import walk_obj


# -- code --
log = logging.getLogger('overmind.server')


class _ResultService:

    def __init__(self):
        self.result: Any = None
        self.exception: Any = None

    def exposed_set_result(self, result):
        self.result = result

    def exposed_set_exception(self, exception):
        self.exception = exception


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
            fn = importlib.import_module(m)
            for a in n.split('.'):
                fn = getattr(fn, a)
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

            master_end, slave_end = multiprocessing.Pipe()
            if pid := os.fork():
                # Master
                try:
                    slave_end.close()
                    from .shmem import hoarder
                    rsvc = _ResultService()
                    # hoarder is needed by filler on slave
                    OneShotServer([hoarder, rsvc], master_end).run()

                    if rsvc.exception:
                        return rsvc.exception

                    if rsvc.result is None:
                        raise ValueError(f'Model {disp} failed to load')

                    model, reuse_key = rsvc.result
                    self._models[key] = model
                    self._models[reuse_key] = model
                finally:
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
            else:
                # Slave
                try:
                    master_end.close()

                    log.info('Forked worker pid = %s', os.getpid())

                    import ctypes
                    ctypes.CDLL(None).prctl(1, 9)

                    from .shmem import filler
                    filler.init_on_slave(slave_end)

                    model, reuse_key = self._load_model(fn, args, kwargs)

                    filler.commit()
                    slave_end.send(('set_result', ((model, reuse_key),), {}))
                    slave_end.recv()
                except Exception as e:
                    log.exception('Error loading model %s', disp)
                    text = traceback.format_exc()
                    try:
                        slave_end.send(('set_exception', (ServiceExceptionInfo(type=type(e), desc=str(e), traceback=text),), {}))
                        slave_end.recv()
                    except:
                        pass
                finally:
                    slave_end.close()
                    os._exit(0)

            self._models_disp.append(disp)

            return self._models[key]

    def _load_model(self, fn, args, kwargs):
        disp = display_of(fn, args, kwargs)
        reuse_key = base64.b32encode(random.randbytes(5)).decode('utf-8')
        b4 = time.time()

        fn, args, kwargs = self._pre_transform((fn, args, kwargs))
        model = fn(*args, **kwargs)
        model = self._post_transform(model)
        try:
            model.__dict__['_overmind_ref'] = OvermindRef(key=reuse_key, disp=disp)
        except AttributeError:
            pass

        log.info('Model %s loaded in %.3fs', disp, time.time() - b4)

        b4 = time.time()
        model = OvermindPickler.dumps(OvermindPickler.dumps(model))  # Double pickle, saving pickle to shared mem too!
        model = bytes(model)
        log.info('Pickled in %.3fs, size = %s bytes', time.time() - b4, len(model))
        return model, reuse_key

    def _pre_transform(self, model):
        from multiprocessing.reduction import ForkingPickler as Pickler

        def pre_transform(m):
            if isinstance(m, OvermindRef):
                m = Pickler.loads(Pickler.loads(self._models[m.key]))
                return False, m
            return True, m

        return walk_obj(model, pre=pre_transform)

    def _post_transform(self, model):
        import torch

        def post_transform(m):
            if not isinstance(m, torch.nn.Module):
                return True, m

            # Remove AlignDevices hooks
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(m, True)

            # Remove accelerate added warning hooks (interferes pickling)
            m.__dict__.pop('to', None)
            m.__dict__.pop('cuda', None)
            m.__dict__.pop('xpu', None)
            m.__dict__.pop('npu', None)

            for p in m.parameters():
                p.requires_grad = False

            return False, m

        return walk_obj(model, pre=post_transform)

    def exposed_shutdown(self):
        log.info('!! Bye')
        os._exit(0)

    def exposed_list_loaded(self):
        return self._models_disp

    def exposed_drop_shell(self):
        import IPython
        IPython.embed()


class BaseServer:

    def __init__(self, services: List[Any]):
        self.services = services

    @staticmethod
    def serve_one(services: List[Any], client: Connection):
        try:
            while True:
                req = client.recv()
                fn = '<unknown>'
                try:
                    fn, args, kwargs = req
                    for svc in reversed(services):
                        f = getattr(svc, f'exposed_{fn}', None)
                        if f is not None:
                            break
                    else:
                        raise AttributeError(f'Function {fn} not found')
                    ret = f(*args, **kwargs)
                except Exception as e:
                    log.exception(f'Error calling {fn}')
                    text = traceback.format_exc()
                    client.send(ServiceExceptionInfo(type=type(e), desc=str(e), traceback=text))
                    continue

                client.send(ret)
        except (EOFError, OSError):
            pass


class OneShotServer(BaseServer):

    def __init__(self, services: List[Any], client: Connection):
        super().__init__(services)
        self.client = client

    def run(self):
        self.serve_one(self.services, self.client)


class ThreadedServer(BaseServer):

    def __init__(self, services: List[Any], listener: Listener):
        super().__init__(services)
        self.listener = listener

    def run(self):
        self.pool = ThreadPool(16)
        while True:
            try:
                client = self.listener.accept()
            except Exception:
                break

            self.pool.apply_async(self.serve_one, [self.services, client])

        self.pool.join()


class NaiveServer(BaseServer):

    def __init__(self, services: List[Any], listener: Listener):
        super().__init__(services)
        self.listener = listener

    def run(self):
        while True:
            try:
                client = self.listener.accept()
            except Exception:
                break

            self.serve_one(self.services, client)



def main():
    import overmind.reducer
    overmind.reducer.init_reductions_server()

    omenv = OvermindEnv.get()
    listener = Listener(omenv.comm_endpoint, authkey=omenv.venv_hash.encode('utf-8'))
    log.info('Overmind server started at %s, pid = %s', omenv.comm_endpoint.replace("\x00", "@"), os.getpid())

    server = NaiveServer([OvermindService()], listener)
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
