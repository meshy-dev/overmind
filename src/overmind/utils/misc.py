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


def walk_obj(obj, pre=None, post=None):
    nop = lambda x: x
    pre = pre or nop
    post = post or nop

    seen = set()
    ref = []

    def walk(m):
        if id(m) in seen:
            return m

        ref.append(m)
        seen.add(id(m))

        recurse, m = pre(m)

        if id(m) not in seen:
            ref.append(m)
            seen.add(id(m))


        if recurse:
            if isinstance(m, type):
                pass
            elif isinstance(m, list):
                for i, v in enumerate(m):
                    m[i] = walk(v)
            elif isinstance(m, tuple):
                m = m.__class__(walk(v) for v in m)
            elif isinstance(m, dict):
                for k, v in m.items():
                    m[k] = walk(v)
            elif (d := getattr(m, '__dict__', None)) is not None:
                for k, v in d.items():
                    d[k] = walk(v)
            elif (keys := getattr(m, '__slots__', None)) is not None:
                for k in keys:
                    setattr(m, k, walk(getattr(m, k)))

        if id(m) not in seen:
            ref.append(m)
            seen.add(id(m))

        m = post(m)

        if id(m) not in seen:
            ref.append(m)
            seen.add(id(m))

        return m

    return walk(obj)
