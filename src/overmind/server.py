# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import argparse
import os
import inspect
import logging
import threading
import multiprocessing.reduction
import time
import types
import uuid

# -- third party --
from frozendict import deepfreeze
from rpyc.utils.server import ThreadedServer
import rpyc
import torch.multiprocessing as mp

# -- own --
from .common import OvermindObjectRef


# -- code --
Pickler = multiprocessing.reduction.ForkingPickler
log = logging.getLogger('overmind.server')


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
        fn, args, kwargs = Pickler.loads(pickled)
        key, disp = self._key_of(fn, args, kwargs)
        if key in self._models:
            log.debug('Providing cached model %s', disp)
            return bytes(Pickler.dumps(self._models[key]))

        with self._loading:
            if key in self._models:
                log.debug('Providing cached model (just loaded!) %s', disp)
                return bytes(Pickler.dumps(self._models[key]))

            args, kwargs = self._convert_refs((args, kwargs))
            log.info('Cold load model %s', disp)
            b4 = time.time()
            model = fn(*args, **kwargs)
            log.info('Model %s loaded in %.3fs', disp, time.time() - b4)
            self._models[key] = model
            rid = str(uuid.uuid4())
            model._overmind_ref = rid
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

    def _key_of(self, fn, args, kwargs):
        s = inspect.signature(fn)
        bs = s.bind(*args, **kwargs)
        bs.apply_defaults()
        args = deepfreeze(bs.arguments)

        if isinstance(fn, types.MethodType):
            if isinstance(fn.__self__, type):
                ty = fn.__self__
            else:
                ty = type(fn.__self__)
            fndisp = f'{ty.__name__}.{fn.__name__}'
        else:
            fndisp = fn.__name__

        args_disp = [f'{k}={repr(v)}' for k, v in bs.arguments.items()]

        disp = f'{fndisp}({", ".join(args_disp)})'

        return (fn, args), disp


def main():
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
    parser.add_argument('--fork', action='store_true')
    options = parser.parse_args()
    from overmind.utils.log import init as init_log

    if options.fork:
        if os.fork():
            return
        os.setsid()
        init_log(logging.DEBUG, '/tmp/overmind.log')
    else:
        init_log(logging.DEBUG, None)

    main()


if __name__ == '__main__':
    start()
