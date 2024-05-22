# -*- coding: utf-8 -*-

# -- stdlib --
from functools import wraps
from typing import Any
import types
import logging

# -- third party --
# -- own --

# -- code --
log = logging.getLogger(__name__)


def hook(target, name=None):
    def inner(hooker):
        funcname = name or hooker.__name__

        hookee: Any = getattr(target, funcname)
        try:
            hookee = object.__getattribute__(target, funcname)
        except AttributeError:
            log.warn(f'@hook: Cannot get raw attr of {target}.{funcname}, fallback to getattr')
            pass

        real_hooker: Any

        if isinstance(hookee, staticmethod):
            hookee = hookee.__func__
            deco = staticmethod

            def static_hooker(*args, **kwargs):
                return hooker(hookee, *args, **kwargs)
            real_hooker = static_hooker

        elif isinstance(hookee, classmethod):
            hookee = hookee.__func__
            deco = classmethod

            def class_hooker(cls, *args, **kwargs):
                return hooker(types.MethodType(hookee, cls), *args, **kwargs)
            real_hooker = class_hooker

        elif isinstance(hookee, types.FunctionType):
            deco = lambda x: x
            if isinstance(target, type):
                def self_hooker(self, *args, **kwargs):
                    return hooker(types.MethodType(hookee, self), *args, **kwargs)
                real_hooker = self_hooker
            else:
                def func_hooker(*args, **kwargs):
                    return hooker(hookee, *args, **kwargs)
                real_hooker = func_hooker
        else:
            raise TypeError(f'Cannot hook {hookee}')

        real_hooker = deco(wraps(hookee)(real_hooker))

        setattr(target, funcname, real_hooker)
        return real_hooker

    return inner
