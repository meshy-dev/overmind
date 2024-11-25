# -*- coding: utf-8 -*-

# -- stdlib --
import sys

# -- third party --
from multiprocessing.connection import Client
from .common import OvermindEnv, ServiceCaller

# -- own --

# -- code --
def _init_client():
    try:
        e = OvermindEnv.get()
        return Client(e.comm_endpoint, authkey=e.venv_hash.encode('utf-8'))
    except Exception:
        print('Overmind seems not started')
        sys.exit(0)


def list_loaded():
    cli = ServiceCaller(_init_client())
    for i in cli.list_loaded():
        print(i)

def drop_shell():
    # for debugging
    cli = _init_client()
    cli.send(('drop_shell', (), {}))


def shutdown():
    cli = _init_client()
    cli.send(('shutdown', (), {}))
