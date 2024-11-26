# -*- coding: utf-8 -*-

# -- stdlib --
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import base64
import dataclasses
import hashlib
import inspect
import multiprocessing.connection
import sys
import tempfile
import types

# -- third party --
# -- own --

# -- code --
IN_OVERMIND_SERVER = None
SERVER_INSTANCE = None


class ServiceException(Exception):
    def __init__(self, info):
        super().__init__()
        self.type = info.type
        self.desc = info.desc
        self.traceback = info.traceback

    def __str__(self):
        return (
            f'Remote server encountered something wrong\n\n'
            '===== Remote Traceback =====\n\n'
        ) +  self.traceback


@dataclass
class ServiceCaller:
    conn: multiprocessing.connection.Connection

    def call(self, fn, *args, **kwargs):
        self.conn.send((fn, args, kwargs))
        ret = self.conn.recv()
        if isinstance(ret, ServiceExceptionInfo):
            raise ret.to_exception()
        return ret

    def __getattr__(self, fn):
        return ServiceInstanceMethod(self, fn)


@dataclass
class ServiceInstanceMethod:
    instance: ServiceCaller
    method: str

    def __call__(self, *args, **kwargs):
        return self.instance.call(self.method, *args, **kwargs)


@dataclass
class ServiceExceptionInfo:
    type: type
    desc: str
    traceback: str

    def to_exception(self):
        return ServiceException(self)


def _deepfreeze(v):
    if isinstance(v, list):
        return tuple(_deepfreeze(i) for i in v)
    elif isinstance(v, dict):
        return tuple(
            (_deepfreeze(k), _deepfreeze(v))
            for k, v in v.items()
        )
    elif hasattr(v, '__dataclass_fields__'):
        return dataclasses.astuple(v)
    else:
        return v


def fqfn_of(fn):
    if isinstance(fn, types.MethodType):
        if isinstance(fn.__self__, type):
            ty = fn.__self__
        else:
            ty = type(fn.__self__)
        fndisp = f'{ty.__module__}.{fn.__qualname__}'
    else:
        fndisp = f'{fn.__module__}.{fn.__qualname__}'

    return fndisp


def display_of(fn, args, kwargs):
    fndisp = fqfn_of(fn)
    args_disp = [repr(v) for v in args]
    kwargs_disp = [f'{k}={repr(v)}' for k, v in kwargs.items()]

    disp = f'{fndisp}({", ".join(args_disp + kwargs_disp)})'
    return disp


def _coalesce_to_kwargs(fn, args, kwargs):
    try:
        s = inspect.signature(fn)
        bs = s.bind(*args, **kwargs)
        kwargs = bs.arguments
    except ValueError:
        kwargs = {**kwargs, '__args': args}

    return kwargs


def key_of(fn, args, kwargs):
    kwargs = _coalesce_to_kwargs(fn, args, kwargs)
    return (fqfn_of(fn), _deepfreeze(kwargs))


@dataclass
class OvermindEnv:
    venv_hash: str
    comm_endpoint: str
    log_path: str
    lock_path: str

    @staticmethod
    @lru_cache(1)
    def get() -> 'OvermindEnv':
        venv_hash = hashlib.sha1(sys.prefix.encode('utf-8')).digest()
        venv_hash = base64.b32encode(venv_hash)[:10].decode('utf-8')

        if sys.platform == 'win32':
            tmpdir = Path(tempfile.gettempdir())
            return OvermindEnv(
                venv_hash=venv_hash,
                comm_endpoint=f'\\\\.\\pipe\\Overmind.{venv_hash}',
                log_path=str(tmpdir / f'overmind.{venv_hash}.log'),
                lock_path='Overmind.Mutex.{venv_hash}',
            )
        else:
            return OvermindEnv(
                venv_hash=venv_hash,
                comm_endpoint=f'\x00overmind.{venv_hash}',
                log_path=f'/tmp/overmind.{venv_hash}.log',
                lock_path=f'/tmp/overmind.{venv_hash}.lock',
            )
