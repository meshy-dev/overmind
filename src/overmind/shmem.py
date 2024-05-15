# -*- coding: utf-8 -*-

# -- stdlib --
from dataclasses import dataclass
from typing import List, Tuple
import base64
import ctypes
import mmap
import os
import random
import threading

# -- third party --
# -- own --
from .common import OvermindEnv


# -- code --
def _make_filename(shift):
    venv = OvermindEnv.get().venv_hash
    rnd = base64.b32encode(random.randbytes(5))
    return f'Overmind-{venv}-{shift}-{rnd}'


SharedMemoryId = Tuple[int, int] | str  # (pid, fd) for (unix), str for shmem name (win32)


class SharedMemory:
    _name = None
    _fd = -1
    _mmap = None
    _buf = None

    _COOKIE = object()

    @classmethod
    def create(cls, shift):
        if not shift > 0:
            raise ValueError("'shift' must be a positive number different from zero")

        if os.name == 'posix':
            return cls._create_posix(shift)
        else:
            return cls._create_win32(shift)

    @classmethod
    def _create_posix(cls, shift):
        libc = ctypes.CDLL(None)
        name = _make_filename(shift).encode('utf-8')
        fd = libc.memfd_create(name, os.O_RDWR)
        os.ftruncate(fd, 1 << shift)
        return cls(fd=fd, name=name, cookie=cls._COOKIE)

    @classmethod
    def _create_win32(cls, shift):
        import _winapi

        map_name = _make_filename(shift)
        h_map = _winapi.CreateFileMapping(
            _winapi.INVALID_HANDLE_VALUE,
            _winapi.NULL,
            _winapi.PAGE_READWRITE,
            ((1 << shift) >> 32) & 0xFFFFFFFF,
            (1 << shift) & 0xFFFFFFFF,
            map_name,
        )
        try:
            return cls(name=map_name, cookie=cls._COOKIE)
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
            h_map = _winapi.OpenFileMapping(_winapi.FILE_MAP_READ, False, name)

            try:
                p_buf = _winapi.MapViewOfFile(h_map, _winapi.FILE_MAP_READ, 0, 0, 0)
            finally:
                _winapi.CloseHandle(h_map)
            size = _winapi.VirtualQuerySize(p_buf)
            self._mmap = mmap.mmap(-1, size, tagname=name)

        self._size = size
        self._buf = memoryview(self._mmap)

    @property
    def view(self):
        assert self._buf
        return self._buf

    @property
    def mem_id(self):
        if os.name == 'posix':
            return os.getpid(), self._fd
        else:
            return self._name

    @classmethod
    def rebuild(cls, mem_id: SharedMemoryId):
        if os.name == 'posix':
            assert isinstance(mem_id, tuple)
            pid, fd = mem_id
            fd = os.open(f'/proc/{pid}/fd/{fd}', os.O_RDWR)
            return cls(fd=fd, cookie=cls._COOKIE)
        else:
            assert isinstance(mem_id, str)
            return cls(name=mem_id, cookie=cls._COOKIE)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._name!r}, size={self._size})'



@dataclass
class Fragment:
    arena: SharedMemoryId
    offset: int
    size: int


@dataclass
class Arena:
    mem: SharedMemory
    current: int


class Hoarder:

    def __init__(self):
        self.shift = 30
        self.arenas: List[Arena] = []
        self.lock = threading.RLock()

    def put(self, data: bytes | memoryview, align=16):
        with self.lock:
            size = len(data)

            for arena in self.arenas:
                current = (arena.current + align - 1) & ~(align - 1)

                if len(arena.mem.view) - current < size:
                    continue

                arena.current = current + size
                memory = arena.mem.view[current:current + size]
                memory[:] = data
                return Fragment(arena=arena.mem.mem_id, offset=current, size=size)
            else:
                self.shift += 1
                self.arenas.append(Arena(
                    mem=SharedMemory.create(self.shift),
                    current=0,
                ))
                return self.put(data, align)


class Borrower:

    def __init__(self):
        self.arenas = {}

    def borrow(self, fragment: Fragment):
        if fragment.arena not in self.arenas:
            self.arenas[fragment.arena] = SharedMemory.rebuild(fragment.arena)

        mem = self.arenas[fragment.arena]
        return mem.view[fragment.offset:fragment.offset + fragment.size].toreadonly()


hoarder = Hoarder()
borrower = Borrower()
