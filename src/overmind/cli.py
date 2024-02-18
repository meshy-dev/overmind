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
