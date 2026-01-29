#include <torch/extension.h>

#include "pybind11/buffer_info.h"
#include "pybind11/gil.h"
#include "pybind11/pytypes.h"

#include <torch/csrc/Storage.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <c10/util/intrusive_ptr.h>

namespace overmind {

struct membuf: std::streambuf {
    membuf(char const* base, size_t size) {
        char* p(const_cast<char*>(base));
        this->setg(p, p, p + size);
    }
    pos_type seekoff(off_type off,
                    std::ios_base::seekdir dir,
                    std::ios_base::openmode which = std::ios_base::in) {
        if (dir == std::ios_base::cur)
            gbump(off);
        else if (dir == std::ios_base::end)
            setg(eback(), egptr() + off, egptr());
        else if (dir == std::ios_base::beg)
            setg(eback(), eback() + off, egptr());
        return gptr() - eback();
    }
    pos_type seekpos(pos_type sp,
                    std::ios_base::openmode which = std::ios_base::in) {
        return seekoff(sp, std::ios_base::beg, which);
    }
};

struct imemstream: virtual membuf, std::istream {
    imemstream(char const* base, size_t size)
        : membuf(base, size)
        , std::istream(static_cast<std::streambuf*>(this)) {
    }
};

inline static uint64_t rotate_right(uint64_t v, unsigned k)
{
    return (v >> k) | (v << (64 - k));
}

inline static uint64_t read_u64(const void * const ptr)
{
    return static_cast<uint64_t>(*reinterpret_cast<const uint64_t*>(ptr));
}

inline static uint64_t read_u32(const void * const ptr)
{
    return static_cast<uint64_t>(*reinterpret_cast<const uint32_t*>(ptr));
}

inline static uint64_t read_u16(const void * const ptr)
{
    return static_cast<uint64_t>(*reinterpret_cast<const uint16_t*>(ptr));
}

inline static uint64_t read_u8 (const void * const ptr)
{
    return static_cast<uint64_t>(*reinterpret_cast<const uint8_t *>(ptr));
}

uint64_t metrohash64_1(const uint8_t * key, uint64_t len, uint32_t seed)
{
    static const uint64_t k0 = 0xC83A91E1;
    static const uint64_t k1 = 0x8648DBDB;
    static const uint64_t k2 = 0x7BDEC03B;
    static const uint64_t k3 = 0x2F5870A5;

    const uint8_t * ptr = reinterpret_cast<const uint8_t*>(key);
    const uint8_t * const end = ptr + len;

    uint64_t hash = ((static_cast<uint64_t>(seed) + k2) * k0) + len;

    if (len >= 32) {
        uint64_t v[4];
        v[0] = hash;
        v[1] = hash;
        v[2] = hash;
        v[3] = hash;

        do {
            v[0] += read_u64(ptr) * k0; ptr += 8; v[0] = rotate_right(v[0],29) + v[2];
            v[1] += read_u64(ptr) * k1; ptr += 8; v[1] = rotate_right(v[1],29) + v[3];
            v[2] += read_u64(ptr) * k2; ptr += 8; v[2] = rotate_right(v[2],29) + v[0];
            v[3] += read_u64(ptr) * k3; ptr += 8; v[3] = rotate_right(v[3],29) + v[1];
        } while (ptr <= (end - 32));

        v[2] ^= rotate_right(((v[0] + v[3]) * k0) + v[1], 33) * k1;
        v[3] ^= rotate_right(((v[1] + v[2]) * k1) + v[0], 33) * k0;
        v[0] ^= rotate_right(((v[0] + v[2]) * k0) + v[3], 33) * k1;
        v[1] ^= rotate_right(((v[1] + v[3]) * k1) + v[2], 33) * k0;
        hash += v[0] ^ v[1];
    }

    if ((end - ptr) >= 16) {
        uint64_t v0 = hash + (read_u64(ptr) * k0); ptr += 8; v0 = rotate_right(v0,33) * k1;
        uint64_t v1 = hash + (read_u64(ptr) * k1); ptr += 8; v1 = rotate_right(v1,33) * k2;
        v0 ^= rotate_right(v0 * k0, 35) + v1;
        v1 ^= rotate_right(v1 * k3, 35) + v0;
        hash += v1;
    }

    if ((end - ptr) >= 8) {
        hash += read_u64(ptr) * k3; ptr += 8;
        hash ^= rotate_right(hash, 33) * k1;
    }

    if ((end - ptr) >= 4) {
        hash += read_u32(ptr) * k3; ptr += 4;
        hash ^= rotate_right(hash, 15) * k1;
    }

    if ((end - ptr) >= 2) {
        hash += read_u16(ptr) * k3; ptr += 2;
        hash ^= rotate_right(hash, 13) * k1;
    }

    if ((end - ptr) >= 1) {
        hash += read_u8 (ptr) * k3;
        hash ^= rotate_right(hash, 25) * k1;
    }

    hash ^= rotate_right(hash, 33);
    hash *= k0;
    hash ^= rotate_right(hash, 33);

    return hash;
}


void initOvermindHelpers(py::module m) {
    m.def("_make_untyped_storage", [](py::buffer b) {
        auto info = new py::buffer_info(b.request());

        if (info->itemsize != 1) throw py::type_error("Buffer item size must be 1");
        if (info->ndim != 1) throw py::type_error("Buffer must be 1-dimensional");
        if (info->format != "B") throw py::type_error("Buffer format must be 'B'");

        auto size = info->size;
        auto ptr = info->ptr;


        return pybind11::reinterpret_steal<py::object>(THPStorage_NewWithStorage(
            THPStorageClass,
            c10::make_intrusive<at::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                size,
                at::DataPtr(
                    ptr,
                    info,
                    [](void* ptr) {
                        py::gil_scoped_acquire gil;
                        auto b = static_cast<py::buffer_info*>(ptr);
                        delete b;
                    },
                    at::DeviceType::CPU
                ),
                /*allocator=*/nullptr,
                /*resizable=*/false)
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 9)
            ,
            c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED
#endif
        ));
    });

    m.def(
        // Copied from torch/csrc/jit/serialization/import.cpp,
        // but accepts a python buffer instead of bytes
        "import_ir_module_from_buffer_0copy",
        [](std::shared_ptr<torch::jit::CompilationUnit> cu, py::buffer buffer) {
            auto info = buffer.request();

            if (info.itemsize != 1) throw py::type_error("Buffer item size must be 1");
            if (info.ndim != 1) throw py::type_error("Buffer must be 1-dimensional");
            if (info.format != "B") throw py::type_error("Buffer format must be 'B'");

            imemstream in((char*)info.ptr, info.size);
            in.seekg(0);

            c10::optional<at::Device> optional_device;
            torch::jit::ExtraFilesMap extra_files_map;

            auto ret = import_ir_module(
                std::move(cu),
                in,
                optional_device,
                extra_files_map,
                /*load_debug_files*/ true,
                /*restore_shapes*/ false);
            return ret;
        }
    );
    m.def(
        "_hash_untyped_storage",
        [](py::handle src) {
            auto UntypedStorage = py::module::import("torch").attr("UntypedStorage");

            if (!py::isinstance(src, UntypedStorage)) {
                throw py::type_error("Source must be an UntypedStorage");
            }

            auto src_storage = reinterpret_cast<THPStorage*>(src.ptr());
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10
            auto src_storage_impl = &src_storage->cdata;
#else
            auto src_storage_impl = src_storage->cdata;
#endif
            auto src_ptr = src_storage_impl->data_ptr().get();
            auto src_size = src_storage_impl->nbytes();
            return metrohash64_1((const uint8_t*)src_ptr, src_size, 233);
        }
    );
    m.def(
        "_hash_buffer",
        [](py::buffer buffer) {
            auto info = buffer.request();

            if (info.itemsize != 1) throw py::type_error("Buffer item size must be 1");
            if (info.ndim != 1) throw py::type_error("Buffer must be 1-dimensional");
            if (info.format != "B") throw py::type_error("Buffer format must be 'B'");

            return metrohash64_1((const uint8_t*)info.ptr, info.size, 233);
        }
    );
    m.def(
        "_memcpy_from_untyped_storage",
        [](py::buffer dst, py::handle src) {
            py::buffer_info dst_info = dst.request();

            if (dst_info.itemsize != 1) throw py::type_error("Buffer item size must be 1");
            if (dst_info.ndim != 1) throw py::type_error("Buffer must be 1-dimensional");
            if (dst_info.format != "B") throw py::type_error("Buffer format must be 'B'");
            if (dst_info.readonly) throw py::value_error("Destination buffer is read-only");

            auto UntypedStorage = py::module::import("torch").attr("UntypedStorage");

            if (!py::isinstance(src, UntypedStorage)) {
                throw py::type_error("Source must be an UntypedStorage");
            }

            auto src_storage = reinterpret_cast<THPStorage*>(src.ptr());
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10
            auto src_storage_impl = &src_storage->cdata;
#else
            auto src_storage_impl = src_storage->cdata;
#endif
            auto src_ptr = src_storage_impl->data_ptr().get();
            auto src_size = src_storage_impl->nbytes();
            if (src_size != (size_t)dst_info.size) {
                throw py::value_error("Source and destination buffers must have the same size");
            }
            std::memcpy(dst_info.ptr, src_ptr, src_size);
        });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  initOvermindHelpers(m);
}


} // namespace overmind
