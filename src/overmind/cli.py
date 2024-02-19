# -*- coding: utf-8 -*-

# -- stdlib --
# -- third party --
import rpyc.utils.factory

# -- own --

# -- code --
def reset():
    conn = rpyc.utils.factory.unix_connect('/tmp/overmind.sock')
    conn.root.reset()
    conn.close()

def list_loaded():
    conn = rpyc.utils.factory.unix_connect('/tmp/overmind.sock')
    l = conn.root.list_loaded()
    for i in l:
        print(i)
