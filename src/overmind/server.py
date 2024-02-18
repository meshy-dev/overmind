# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import argparse
import inspect
import logging
import multiprocessing.reduction
import pickle
import time
import types

# -- third party --
from daemonize import Daemonize
from frozendict import deepfreeze
from rpyc.utils.server import ThreadedServer
import rpyc
import torch.multiprocessing as mp

# -- own --

# -- code --
Pickler = multiprocessing.reduction.ForkingPickler
log = logging.getLogger('overmind.server')


class OvermindService(rpyc.Service):

    def __init__(self):
        super().__init__()
        self._models = {}

    def exposed_ping(self):
        return 'pong'

    def exposed_load(self, pickled):
        fn, args, kwargs = pickle.loads(pickled)
        key, disp = self._key_of(fn, args, kwargs)
        if key in self._models:
            log.debug('Providing cached model %s', disp)
            return bytes(Pickler.dumps(self._models[key]))

        assert key not in self._models

        log.info('Cold load model %s', disp)
        b4 = time.time()
        model = fn(*args, **kwargs)
        log.info('Model %s loaded in %.3fs', disp, time.time() - b4)
        self._models[key] = model
        return bytes(Pickler.dumps(self._models[key]))

    def exposed_reset(self):
        log.info('!! Reset cache')
        self._models.clear()

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


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument('--daemon', action='store_true')
    options = parser.parse_args()

    from overmind.utils.log import init as init_log

    if options.daemon:
        init_log(logging.INFO, '/tmp/overmind.log')
        daemon = Daemonize(app="overmind", pid="/tmp/overmind.pid", action=main, logger=logging.getLogger('daemonize'))
        daemon.start()
    else:
        init_log(logging.DEBUG, None)
        main()

if __name__ == '__main__':
    start()
