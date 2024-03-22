# -*- coding: utf-8 -*-

# -- stdlib --
import sys

# -- third party --
from multiprocessing.connection import Client
from .common import OvermindEnv

# -- own --

# -- code --
def _init_client():
    try:
       return Client(OvermindEnv.get().comm_endpoint)
    except Exception:
        print('Overmind seems not started')
        sys.exit(0)


def reset():
    cli = _init_client()
    cli.send(('reset', (), {}))


def list_loaded():
    cli = _init_client()
    cli.send(('list_loaded', (), {}))
    l = cli.recv()
    for i in l:
        print(i)


def shutdown():
    cli = _init_client()
    cli.send(('shutdown', (), {}))
