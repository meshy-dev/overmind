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
        comm = OvermindEnv.get().comm_endpoint
        return Client(comm, authkey=comm.encode('utf-8'))
    except Exception:
        print('Overmind seems not started')
        sys.exit(0)


def list_loaded():
    cli = _init_client()
    cli.send(('list_loaded', (), {}))
    l = cli.recv()
    for i in l:
        print(i)


def shutdown():
    cli = _init_client()
    cli.send(('shutdown', (), {}))
