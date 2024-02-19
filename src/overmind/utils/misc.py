# -*- coding: utf-8 -*-

# -- stdlib --
from functools import wraps
import types

# -- third party --
# -- own --

# -- code --
def hook(module, name=None):
    def inner(hooker):
        funcname = name or hooker.__name__
        hookee = getattr(module, funcname)

        if isinstance(hookee, types.MethodType) and isinstance(hookee.__self__, type):
            hookee = hookee.__func__
            @classmethod
            @wraps(hookee)
            def real_hooker(cls, *args, **kwargs):
                return hooker(types.MethodType(hookee, cls), *args, **kwargs)

        else:
            @wraps(hookee)
            def real_hooker(*args, **kwargs):
                return hooker(hookee, *args, **kwargs)

        try:
            real_hooker.orig = hookee
        except AttributeError:
            pass

        setattr(module, funcname, real_hooker)
        return real_hooker

    return inner
