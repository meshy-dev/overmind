# -*- coding: utf-8 -*-

# -- stdlib --
from dataclasses import dataclass
from typing import List, TYPE_CHECKING, Tuple, Dict
import base64
import logging
import multiprocessing
import ctypes
import mmap
import os
import random
import threading

# -- third party --
# -- own --
from .common import OvermindEnv

# -- typing --
if TYPE_CHECKING:
    from torch import UntypedStorage  # noqa: F401


# -- code --
log = logging.getLogger('overmind.shmem')


def _make_filename(shift):
    venv = OvermindEnv.get().venv_hash
    rnd = base64.b32encode(random.randbytes(5)).decode('utf-8')
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
class ExportedArena:
    tag: int
    mem_id: SharedMemoryId
    size: int
    current: int


@dataclass
class Arena:
    tag: int
    mem: SharedMemory
    current: int


class Hoarder:

    def __init__(self):
        self.shift = 34
        self.arenas: Dict[int, Arena] = {}
        self.fragments: Dict[int, Fragment] = {}
        self.lock = threading.RLock()
        self.master_conn, self.slave_conn = multiprocessing.Pipe()

    def exposed_allocate(self) -> ExportedArena | None:
        with self.lock:
            self.shift += 1
            log.debug('Hoarder: Creating new arena with shift = %s', self.shift)
            tag = random.getrandbits(24)
            arena = Arena(
                tag=tag,
                mem=SharedMemory.create(self.shift),
                current=0,
            )
            self.arenas[tag] = arena
            return ExportedArena(
                tag=arena.tag,
                mem_id=arena.mem.mem_id,
                size=len(arena.mem.view),
                current=arena.current,
            )

    def exposed_merge(self, latest_ptrs: List[Tuple[int, int]], fragments: Dict[int, Fragment]):
        with self.lock:
            for tag, current in latest_ptrs:
                self.arenas[tag].current = max(self.arenas[tag].current, current)

            self.fragments.update(fragments)


class Filler:
    # Runs on forked slave

    def __init__(self):
        self.synchronized = False
        self.arenas: Dict[int, Arena] = {}
        self.fragments: Dict[int, Fragment] = {}
        self.new_fragments: Dict[int, Fragment] = {}
        self.lock = threading.RLock()

    def synchronize(self):
        assert not self.synchronized
        self.synchronized = True
        self.arenas = hoarder.arenas
        self.fragments = hoarder.fragments
        hoarder.master_conn.close()

    def request_arena(self):
        hoarder.slave_conn.send(('allocate', (), {}))
        arena = hoarder.slave_conn.recv()
        log.debug('Filler: got new arena %s', arena)
        self.arenas[arena.tag] = Arena(
            tag=arena.tag,
            mem=SharedMemory.rebuild(arena.mem_id),
            current=0,
        )

    def commit(self):
        assert self.synchronized
        with self.lock:
            hoarder.slave_conn.send(('merge', ([arena.tag, arena.current] for arena in self.arenas.values()), self.new_fragments))
            self.new_fragments.clear()

    def put(self, data: 'bytes | memoryview | UntypedStorage', align=16):
        assert self.synchronized
        import overmind._C
        from torch import UntypedStorage

        if isinstance(data, UntypedStorage):
            digest = overmind._C._hash_untyped_storage(data)
        else:
            digest = overmind._C._hash_buffer(data)

        if digest in self.fragments:
            return self.fragments[digest]

        with self.lock:
            size = len(data)

            for tag, arena in self.arenas.items():
                current = (arena.current + align - 1) & ~(align - 1)

                if current + size > len(arena.mem.view):
                    continue

                arena.current = current + size
                memory = arena.mem.view[current:current + size]
                if isinstance(data, UntypedStorage):
                    # Special method to speed up things
                    assert data.device.type == 'cpu'
                    overmind._C._memcpy_from_untyped_storage(memory, data)
                else:
                    memory[:] = data
                frag = Fragment(arena=arena.mem.mem_id, offset=current, size=size)
                self.fragments[digest] = frag
                self.new_fragments[digest] = frag
                return frag
            else:
                self.request_arena()
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
filler = Filler()
borrower = Borrower()
