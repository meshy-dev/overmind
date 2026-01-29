# -*- coding: utf-8 -*-

# -- stdlib --
from multiprocessing.connection import Connection, Listener
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List
import argparse
import base64
import importlib
import logging
import multiprocessing
import os
import random
import sys
import threading
import time
import traceback

# -- third party --
# -- own --
from .common import OvermindEnv, ServiceCaller, ServiceExceptionInfo, display_of
from .reducer import OvermindPickler, OvermindRef
from .utils.misc import walk_obj


# -- code --
log = logging.getLogger('overmind.server')


class OvermindService:

    def __init__(self):
        super().__init__()
        self._models: Dict[str, bytes] = {}
        self._models_disp: List[str] = []  # solely for debugging
        self._loading = threading.Lock()

    def ping(self):
        return 'pong'

    def load(self, fnspec, args, kwargs, key, disp):
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
                    kwargs['device_map'] = 'cuda:0'
                else:
                    log.warning('Auto device_map is not supported, forcing cpu')
                    kwargs.pop('device_map')
                    kwargs['device_map'] = 'cpu'
                break

            elif isinstance(dmap, dict) and len(dmap) == 1:
                dev = next(iter(dmap.values()))

                if dev in ('cpu', 'cuda', torch.device('cpu')):
                    break

                if isinstance(dev, torch.device) and dev.type == 'cuda':
                    if dev.index not in (None, 0):
                        raise ValueError(f'Only models on cuda:0 are supported, not loading {disp}')

                    kwargs['device_map'] = {'': torch.device('cuda:0')}

            elif dmap in ('cpu', 'cuda', 'cuda:0'):
                pass

            else:
                raise ValueError('Complex device_map is not supported')

            break
        # End of heuristics

        from .shmem import hoarder

        if key in self._models:
            payload = self._models[key]
            log.debug('Providing cached model %s (%s bytes over wire)', disp, len(payload))
            return hoarder.export_arenas(), payload

        with self._loading:
            if key in self._models:
                log.debug('Providing cached model (just loaded!) %s', disp)
                return hoarder.export_arenas(), self._models[key]

            log.info('Cold load model %s', disp)

            master_end, slave_end = multiprocessing.Pipe()
            slave_proc = multiprocessing.Process(target=self._slave, args=(slave_end,))
            try:
                slave_proc.start()
                slave = ServiceCaller(master_end)
                slave.push_state(hoarder.export(), self._models)
                model, reuse_key = slave.load_model(fnspec, args, kwargs)
                model = bytes(model)
                hoarder_state = slave.pull_state()
                hoarder.merge(hoarder_state, True)
                self._models[key] = model
                self._models[reuse_key] = model
                self._models_disp.append(f'{reuse_key} {disp}')
                return hoarder.export_arenas(), self._models[key]
            finally:
                slave_proc.kill()
                slave_proc.join()

    @classmethod
    def _slave(cls, conn):
        init_log()

        log.info('Spawned worker pid = %s', os.getpid())

        import ctypes
        ctypes.CDLL(None).prctl(1, 9)

        from .reducer import init_reductions_server
        init_reductions_server()

        from .shmem import ExportedHoarder

        class SlaveService:

            def push_state(self, hoarder_state: ExportedHoarder, existing_models: Dict[str, bytes]):
                from .shmem import hoarder
                hoarder.merge(hoarder_state, False)
                self._models = existing_models

            def pull_state(self):
                from .shmem import hoarder
                return hoarder.export()

            def load_model(self, fnspec, args, kwargs):
                from .shmem import borrower
                borrower.link_to_hoarder()

                from .reducer import OvermindUnpickler
                fnspec = OvermindUnpickler.loads(fnspec)

                if isinstance(fnspec, tuple):
                    # This makes pickle happy
                    m, n = fnspec
                    fn = importlib.import_module(m)
                    for a in n.split('.'):
                        fn = getattr(fn, a)
                else:
                    fn = fnspec

                disp = display_of(fn, args, kwargs)
                reuse_key = base64.b32encode(random.randbytes(5)).decode('utf-8')
                b4 = time.time()

                fn, args, kwargs = cls._pre_transform((fn, args, kwargs), self._models)
                model = fn(*args, **kwargs)
                model = cls._post_transform(model)
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

        server = OneShotServer([SlaveService()], conn)
        server.run()

    @staticmethod
    def _pre_transform(model, existing_models):
        from multiprocessing.reduction import ForkingPickler as Pickler

        def pre_transform(m):
            if isinstance(m, OvermindRef):
                m = Pickler.loads(Pickler.loads(existing_models[m.key]))
                return False, m
            return True, m

        return walk_obj(model, pre=pre_transform)

    @staticmethod
    def _post_transform(model):
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

        model = walk_obj(model, pre=post_transform)
        return model

    def shutdown(self):
        log.info('!! Bye')
        os._exit(0)

    def list_loaded(self):
        return self._models_disp

    def drop_shell(self):
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
                    if fn.startswith('_'):
                        raise AttributeError(f'Function {fn} not found')

                    for svc in reversed(services):
                        f = getattr(svc, fn, None)
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

    def stop(self):
        pass


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

    def stop(self):
        self.listener.close()


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

    def stop(self):
        self.listener.close()



def main():
    from . import common
    multiprocessing.set_start_method('spawn')

    # CUDA sentinels to prevent CUDA initialization at master
    import torch._C
    torch._C._cuda_init = ('_real', torch._C._cuda_init)
    torch._C._cuda_getDeviceCount = ('_real', torch._C._cuda_getDeviceCount)
    torch._C._cuda_getArchFlags = ('_real', torch._C._cuda_getArchFlags)

    omenv = OvermindEnv.get()
    listener = Listener(omenv.comm_endpoint, authkey=omenv.venv_hash.encode('utf-8'))
    log.info('Overmind server started at %s, pid = %s', omenv.comm_endpoint.replace("\x00", "@"), os.getpid())

    server = ThreadedServer([OvermindService()], listener)
    assert common.SERVER_INSTANCE is None
    common.SERVER_INSTANCE = server
    server.run()


def init_log():
    from overmind.utils.log import init as init_log
    omenv = OvermindEnv.get()
    init_log(logging.DEBUG, f'/tmp/overmind.{omenv.venv_hash}.log')


def daemon_main():
    init_log()
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
