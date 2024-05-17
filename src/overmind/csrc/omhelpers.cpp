#include <torch/extension.h>

#include "pybind11/buffer_info.h"
#include "pybind11/pytypes.h"

/* #include <ATen/core/dispatch/OperatorOptions.h> */
/* #include <ATen/core/ivalue.h> */
/* #include <ATen/core/stack.h> */
/* #include <torch/csrc/MemoryFormat.h> */
#include <torch/csrc/Storage.h>
/* #include <torch/csrc/jit/ir/ir.h> */
/* #include <torch/csrc/jit/jit_log.h> */
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

void initOvermindHelpers(py::module m) {
    m.def("_make_untyped_storage", [](py::buffer b) {
        py::buffer_info info = b.request();

        if (info.itemsize != 1) throw py::type_error("Buffer item size must be 1");
        if (info.ndim != 1) throw py::type_error("Buffer must be 1-dimensional");
        if (info.format != "B") throw py::type_error("Buffer format must be 'B'");

        auto size = info.size;
        auto ptr = info.ptr;

        return py::handle(THPStorage_New(
            c10::make_intrusive<at::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                size,
                at::DataPtr(
                    ptr,
                    new py::buffer_info(std::move(info)),
                    [](void* ptr) {
                        auto b = static_cast<py::buffer_info*>(ptr);
                        delete b;
                    },
                    at::DeviceType::CPU
                ),
                /*allocator=*/nullptr,
                /*resizable=*/false)
            )
        );
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
            auto src_storage_impl = src_storage->cdata;
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
