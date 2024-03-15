# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import argparse
import logging
import multiprocessing.reduction
import importlib
import dataclasses
import os
import threading
import time
import types
import uuid

# -- third party --
from daemonize import Daemonize
from rpyc.utils.server import ThreadedServer
import rpyc.core.protocol

# -- own --
from .common import OvermindObjectRef, key_of, display_of


# -- code --
Pickler = multiprocessing.reduction.ForkingPickler
log = logging.getLogger('overmind.server')
rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout'] = 900


class OvermindService(rpyc.Service):

    def __init__(self):
        super().__init__()
        self._models = {}
        self._models_byref = {}
        self._models_disp = []  # solely for debugging
        self._loading = threading.Lock()

    def exposed_ping(self):
        return 'pong'

    def exposed_load(self, pickled):
        import torch

        v, kwargs = Pickler.loads(pickled)
        if isinstance(v, tuple):
            # This makes pickle happy
            m, n = v
            fn = getattr(importlib.import_module(m), n)
        else:
            fn = v

        key = key_of(fn, kwargs)
        disp = display_of(fn, kwargs)

        # Heuristics:
        if 'device' in kwargs and kwargs.get('device') not in ('cpu', torch.device('cpu')):
            log.error('Not caching model %s because it is not on CPU', disp)
            raise ValueError(f'Not caching model {disp} because it is not on CPU')
        # End of heuristics

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
            model = fn(**kwargs)
            log.info('Model %s loaded in %.3fs', disp, time.time() - b4)
            self._models[key] = model
            rid = str(uuid.uuid4())

            try:
                model._overmind_ref = rid
            except AttributeError:
                pass

            self._models_byref[rid] = model
            self._models_disp.append(disp)
            data = bytes(Pickler.dumps(self._models[key]))
            log.info(f'Will send {len(data)} bytes')
            return data

    def exposed_reset(self):
        log.info('!! Reset cache')
        self._models.clear()
        self._models_byref.clear()
        self._models_disp.clear()

    def exposed_list_loaded(self):
        return self._models_disp

    def _convert_refs(self, obj):
        if isinstance(obj, (list, tuple)):
            return obj.__class__([self._convert_refs(x) for x in obj])
        elif isinstance(obj, dict):
            return {k: self._convert_refs(v) for k, v in obj.items()}
        elif isinstance(obj, OvermindObjectRef):
            return self._models_byref[str(obj)]
        else:
            return obj


def main():
    import torch.multiprocessing as mp

    mp.set_sharing_strategy('file_system')
    sock = Path('/tmp/overmind.sock')
    if sock.exists():
        sock.unlink()

    server = ThreadedServer(OvermindService(), socket_path=str(sock), logger=logging.getLogger('rpyc'))
    server.start()


def daemon_main():
    from overmind.utils.log import init as init_log
    init_log(logging.DEBUG, '/tmp/overmind.log')
    main()


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument('--daemon', action='store_true')
    parser.add_argument('--fork', action='store_true')
    options = parser.parse_args()
    from overmind.utils.log import init as init_log

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if options.daemon:
        pid = Path('/tmp/overmind.pid')
        if pid.exists():
            pid.unlink()
        daemon = Daemonize(app="overmind", pid="/tmp/overmind.pid", action=daemon_main, logger=logging.getLogger('daemonize'))
        daemon.start()
    elif options.fork:
        if os.fork():
            return
        os.setsid()
        init_log(logging.DEBUG, '/tmp/overmind.log')
        main()
    else:
        init_log(logging.DEBUG, None)
        main()


if __name__ == '__main__':
    start()
