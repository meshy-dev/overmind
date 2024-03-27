# -*- coding: utf-8 -*-

"""Provides shared memory for direct access across processes.

The API of this package is currently provisional. Refer to the
documentation for details.


Modified from multiprocessing.shared_memory.SharedMemory
"""

# -- stdlib --
import base64
import mmap
import ctypes
import os
import random

# -- third party --
# -- own --
from .common import OvermindEnv


# -- code --
__all__ = [ 'SharedMemory' ]


def _make_filename():
    venv = OvermindEnv.get().venv_hash
    rnd = base64.b32encode(random.randbytes(5))
    return f'Overmind-{venv}-{rnd}'


class SharedMemory:
    _name = None
    _fd = -1
    _mmap = None
    _buf = None

    _COOKIE = object()

    @classmethod
    def create(cls, size):
        if not size > 0:
            raise ValueError("'size' must be a positive number different from zero")

        if os.name == 'posix':
            return cls._create_posix(size)
        else:
            return cls._create_win32(size)

    @classmethod
    def _create_posix(cls, size):
        libc = ctypes.CDLL(None)
        fd = libc.memfd_create(b'overmind-shmem', os.O_RDWR)
        os.ftruncate(fd, size)
        return cls(fd=fd, cookie=cls._COOKIE)

    @classmethod
    def _create_win32(cls, size):
        import _winapi
        while True:
            temp_name = _make_filename()
            # Create and reserve shared memory block with this name
            # until it can be attached to by mmap.
            h_map = _winapi.CreateFileMapping(
                _winapi.INVALID_HANDLE_VALUE,
                _winapi.NULL,
                _winapi.PAGE_READWRITE,
                (size >> 32) & 0xFFFFFFFF,
                size & 0xFFFFFFFF,
                temp_name
            )
            try:
                last_error_code = _winapi.GetLastError()
                if last_error_code == _winapi.ERROR_ALREADY_EXISTS:
                    continue
                return cls(name=temp_name, cookie=cls._COOKIE)
            finally:
                _winapi.CloseHandle(h_map)

    def __init__(self, fd=None, name=None, cookie=None):
        if cookie is not self._COOKIE:
            raise Exception('Use SharedMemory.create!')

        if os.name == 'posix':
            assert fd
            self._name = name or 'memfd:overmind-shmem'
            self._fd = fd
            stats = os.fstat(self._fd)
            size = stats.st_size
            self._mmap = mmap.mmap(self._fd, size)
        else:
            assert name
            self._name = name
            import _winapi
            # Dynamically determine the existing named shared memory
            # block's size which is likely a multiple of mmap.PAGESIZE.
            h_map = _winapi.OpenFileMapping(
                _winapi.FILE_MAP_READ,
                False,
                name
            )

            try:
                p_buf = _winapi.MapViewOfFile(
                    h_map,
                    _winapi.FILE_MAP_READ,
                    0,
                    0,
                    0
                )
            finally:
                _winapi.CloseHandle(h_map)
            size = _winapi.VirtualQuerySize(p_buf)
            self._mmap = mmap.mmap(-1, size, tagname=name)

        self._size = size
        self._buf = memoryview(self._mmap)

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass

    def __reduce__(self):
        if os.name == 'posix':
            return (
                self._rebuild_fd,
                (os.getpid(), self._fd),
            )
        else:
            return (
                self._rebuild_win32,
                (self._name, ),
            )

    @classmethod
    def _rebuild_fd(cls, pid, fd):
        fd = os.open(f'/proc/{pid}/fd/{fd}', os.O_RDWR)
        return cls(fd=fd, cookie=cls._COOKIE)

    @classmethod
    def _rebuild_win32(cls, name):
        return cls(name=name, cookie=cls._COOKIE)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r}, size={self.size})'

    @property
    def buf(self):
        "A memoryview of contents of the shared memory block."
        return self._buf

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        "Size in bytes."
        return self._size

    def close(self):
        """Closes access to the shared memory from this instance but does
        not destroy the shared memory block."""
        self._buf = None
        self._mmap = None
        if os.name == 'posix' and self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
