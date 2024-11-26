# -*- coding: utf-8 -*-

# -- stdlib --
from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING, Tuple
import base64
import ctypes
import logging
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
    _mem_id = None
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
        mem_id = (os.getpid(), fd)
        return cls(fd=fd, name=name, mem_id=mem_id, cookie=cls._COOKIE)

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

    def __init__(self, fd=None, name=None, mem_id=None, cookie=None):
        if cookie is not self._COOKIE:
            raise Exception('Use SharedMemory.create!')

        if os.name == 'posix':
            assert fd
            self._name = name or 'memfd:overmind-shmem'
            self._fd = fd
            self._mem_id = mem_id
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
    def mem_id(self) -> SharedMemoryId:
        if os.name == 'posix':
            return self._mem_id
        else:
            return self._name

    def to_owned(self):
        if os.name == 'posix':
            self._mem_id = (os.getpid(), self._fd)
        return self

    @classmethod
    def rebuild(cls, mem_id: SharedMemoryId):
        if os.name == 'posix':
            assert isinstance(mem_id, tuple)
            pid, fd = mem_id
            local_fd = os.open(f'/proc/{pid}/fd/{fd}', os.O_RDWR)
            return cls(fd=local_fd, mem_id=mem_id, cookie=cls._COOKIE)
        else:
            assert isinstance(mem_id, str)
            return cls(name=mem_id, cookie=cls._COOKIE)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._name!r}, size={self._size})'


ArenaTag = int
ContentHash = int


@dataclass
class ExportedArena:
    tag: ArenaTag
    mem_id: SharedMemoryId


@dataclass
class Fragment:
    digest: ContentHash
    arena: ArenaTag
    offset: int
    size: int


@dataclass
class Arena:
    tag: ArenaTag
    mem: SharedMemory
    current: int


@dataclass
class ExportedHoarder:
    shift: int
    arenas: List[ExportedArena]
    fragments: List[Fragment]


class Hoarder:

    def __init__(self):
        self.shift = 33
        self.arenas: Dict[ArenaTag, Arena] = {}
        self.fragments: Dict[ContentHash, Fragment] = {}
        self.lock = threading.RLock()

    def allocate(self):
        with self.lock:
            self.shift += 1
            log.debug('Hoarder: Creating new arena with shift = %s', self.shift)
            tag = random.getrandbits(48)
            arena = Arena(
                tag=tag,
                mem=SharedMemory.create(self.shift),
                current=0,
            )
            self.arenas[tag] = arena

    def export_arenas(self) -> List[ExportedArena]:
        return [ExportedArena(tag=arena.tag, mem_id=arena.mem.mem_id) for arena in self.arenas.values()]

    def export(self) -> ExportedHoarder:
        arenas = self.export_arenas()
        return ExportedHoarder(
            shift=self.shift,
            arenas=arenas,
            fragments=list(self.fragments.values()),
        )

    def merge(self, exported: ExportedHoarder, take_ownership: bool):
        with self.lock:
            self.shift = exported.shift

            for a in exported.arenas:
                if a.tag not in self.arenas:
                    mem = SharedMemory.rebuild(a.mem_id)
                    if take_ownership:
                        mem.to_owned()
                    self.arenas[a.tag] = Arena(tag=a.tag, mem=mem, current=0)

            for frag in exported.fragments:
                if frag.digest in self.fragments:
                    continue

                tag = frag.arena
                assert tag in self.arenas
                arena = self.arenas[tag]
                current = frag.offset + frag.size
                arena.current = max(arena.current, current)

            self.fragments.update({frag.digest: frag for frag in exported.fragments})

    def put(self, data: 'bytes | memoryview | UntypedStorage', align=16):
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

            for arena in self.arenas.values():
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
                frag = Fragment(digest=digest, arena=arena.tag, offset=current, size=size)
                self.fragments[digest] = frag
                return frag
            else:
                self.allocate()
                return self.put(data, align)


class Borrower:

    def __init__(self):
        self.arenas: Dict[ArenaTag, Arena] = {}

    def import_arenas(self, arenas: List[ExportedArena]):
        for a in arenas:
            if a.tag not in self.arenas:
                self.arenas[a.tag] = Arena(
                    tag=a.tag,
                    mem=SharedMemory.rebuild(a.mem_id),
                    current=0,  # doesn't matter here
                )

    def link_to_hoarder(self):
        self.arenas = hoarder.arenas

    def borrow(self, fragment: Fragment):
        mem = self.arenas[fragment.arena].mem
        return mem.view[fragment.offset:fragment.offset + fragment.size].toreadonly()


hoarder = Hoarder()
borrower = Borrower()
