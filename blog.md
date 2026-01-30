# Overmind: A non-intrusive method/library to cut model loading time from 15s -> 0.2s

## Rationale

It all begins 2 years ago, when we shipped our first try of lowpoly generation mode. The lowpoly mode did not go well, it emits poor results from today's perspective, but we paid a lot for it -- a dedicated GPU only processes single digit tasks per day. It has fine-tuned weights, big enough to drive all other model weights out of VRAM. Worse, we have maybe 3 such modes (can't remember the exact number), they constituted a significant part of our inference infra, made a quite unforgiving efficiency ratio. And no, we can't naively load the models just-in-time, it costs 30s, larger than the actual processing time.

We don't have dedicated pipeline engineers then, our algorithm devs tried their best to workaround this. Days later, our codebase was littered with `this.to('cpu')` and `that.to('cuda')`. This approach works for a while, but break the flow of our algo devs from time to time. What if things can happen automagically? It's Python, things do happen automagically in Python.

## How do you define 'automagically' ?

Let's jump into the role of an algorithm developer. Things are pretty clear: I don't want to care about the runtime performance outside my core algorithm unless I absolutely have to. I would rather not know anything about model swap in and out.

Of course we can't achieve that, but we can try minimizing the intrusion we have to introduce to algorithm code. This reminds me the monkey-patching of `gevent` library, it patches (primarily) the `socket` library, replaces it with `gevent.socket` which can switch to other greenlets when IO would block, much like a goroutine (actually `gevent` is older than Golang!).

Since we were only using HuggingFace libs (`transformers`, `diffusers`) to load models at the time, the target become clear: We only introduce a monkey-patch call, and the rest of code should remain unchanged, `XXXPipeline.from_pretrained(...)` should be much faster.

## Some Facts, Obvious Decisions and Assumptions

**Overmind is a caching library, it caches model loading call results into system memory and later reconstruct it fast.**

We skip discussing about how monkey-patching is implemented, that's a not-so-interesting detail. All we need to know is, it redirects all the `XXXPipeline.from_pretrained(...)` calls to `overmind.api.load(XXXPipeline.from_pretrained, ...)`.

We use `pickle` to serialize our cache result since... we have no choice, and `torch.save` itself uses `pickle`, it's weird not to use it.

We use a client/server architecture since we don't want to invalidate our cache when process terminates. There are many subprocess calls could benefit from it.

We assume `XXXPipeline.from_pretrained` parameters to be simple hashable things (`str` and things alike) and other models loaded by `overmind` (explained later).

The name `overmind` is borrowed from Starcraft, as you may have guessed.

## Reconstruct it fast!

We can't naively save `pickle.loads` result in memory and call it a day. After all, on a warmed up scenario, Linux page cache did its job caching on-disk models and we can still see a loading time measured in tens of seconds.

The inefficiency comes from memory copying. In Python, even creating millions of objects would cost no more than several hundred ms. However, for a memory copy of 10GiB, it would cost half a second. We must avoid memory copy as much as possible.

Fortunately, most of the big memory chunks are Torch tensors, we can safely address only them and ignore the rest.

Actually, I got the knowledge of the internal structure of a Torch tensor in the reduction code while researching the tensor sharing mechanism:

```python
# Copied from torch.multiprocessing.reductions, most of the code is removed
def reduce_tensor(tensor):
    ...
    storage = tensor._typed_storage()
    ...
    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        tensor.requires_grad,
    )
    return (rebuild_tensor, (type(tensor), storage, metadata))
```

Quite simple: a tensor is its type, its metadata and its underlying storage. Here `storage` is of type `TypedStorage`, but actually `TypedStorage` is just a simple wrapper to `UntypedStorage`. `UntypedStorage` is the class that actually holding all the tensor data.

Our task become more specific now: How do we avoid copying `UntypedStorage`? Can we manage these tensor memory by ourselves and construct `UntypedStorage`s by pointing to the memory we manage?

The answer is yes!

Skimming through the C++ code of where `UntypedStorage` is constructed, we can easily find a code snippet like this:

```cpp
// Copied from torch/csrc/Storage.cpp
static PyObject* THPStorage_get(THPStorage* self, PyObject* index) {
    // ...omitting unrelated code...

    auto new_storage_impl = make_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        slicelength,
        at::DataPtr(
            static_cast<void*>(data + start),
            old_storage_impl,
            [](void* s) {
              c10::raw::intrusive_ptr::decref(static_cast<at::StorageImpl*>(s));
            },
            old_storage_impl->device()),
        old_storage_impl->allocator(),
        /* resizable */ false,
        device_opt);

    PyObject* _ret =
        THPStorage_NewWithStorage(Py_TYPE(self), std::move(new_storage_impl));

    return _ret;
}
```

Not only can we use a pointer, but the `at::DataPtr` class can also handle destruction, making the lifetime management much simpler.

On the Python side, a pointer to a memory region is represented by a `memoryview` object, these objects support buffer protocol. We can get a `memoryview` object from many things, `bytes` and `mmap` are the 2 major things supporting it, and they are also what we care about.

Finally, we know what we should do: create a function that accepts a `memoryview` object and turns it into an `UntypedStorage` without copying. With ability to reconstruct `UntypedStorage` from `memoryview`, the actual tensor data don't have to be in the pickle stream, greatly reduced the data size we have to copy around.


```cpp
void initOvermindHelpers(py::module m) {
    // ...
    m.def("_make_untyped_storage", [](py::buffer b) {
        auto info = new py::buffer_info(b.request());

        return pybind11::reinterpret_steal<py::object>(THPStorage_NewWithStorage(
            THPStorageClass,
            c10::make_intrusive<at::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                info->size,
                at::DataPtr(
                    info->ptr, info,
                    [](void* ptr) {
                        py::gil_scoped_acquire gil;
                        auto b = static_cast<py::buffer_info*>(ptr);
                        delete b;
                    },
                    at::DeviceType::CPU
                ),
                /*allocator=*/nullptr,
                /*resizable=*/false,
            )
        ));
    });
}
```

That's the core building block of `overmind`.


## Sharing the tensors!


> **Note:** There's already a tensor sharing mechanism in PyTorch, but it doesn't fit our needs. More on this later.


### First, sharing memory between client and server

When we see 'share' and 'memory' comes together, we all have an urge to use `shmget` and its friends. It is "designed" to be used as a memory sharing mechanism, right? But it has 2 major flaws:

- POSIX shm is a scarce resource, what you can use is determined by how sysadmin configure the system. An extreme but ubiquitous example is Docker containers, by default you have only 64MiB POSIX shm usable.
- POSIX shm lives longer than your process, you have to do your own management. If the management process is forcefully killed, or didn't handle it carefully, the shm object could be left on the system indefinitely.

If you look into carefully, Linux is full of interesting system calls. `memfd_create` is one we are interested: It gives you an fd represents an allocation of anonymous memory. You can do all sorts of file operations on it: read, write, and, of course, mmap. If we can share the fd, we can share the memory.

Sharing an fd has a 'standard' but arcane way to do it: `sendmsg` with `SCM_RIGHTS`. We can leverage libraries to help us hide the daunting details of the `sendmsg` process, but we still have to do our coordination between server and client processes. We decided to use a hack here: Just open `/proc/{pidof(server)}/fd/{memfd}` on the client side, while never closing the fd on the `overmind` server side. The only communication needed is a `(pid, fd)` tuple. It works perfectly in our case.

The above words boils down to these lines:

```python
class SharedMemory:
    @classmethod
    def create(cls, shift):
        # Called on server side
        libc = ctypes.CDLL(None)
        name = _make_filename(shift).encode('utf-8')
        fd = libc.memfd_create(name, os.O_RDWR)
        os.ftruncate(fd, 1 << shift)
        mem_id = (os.getpid(), fd)
        return cls(fd=fd, mem_id=mem_id)

    @classmethod
    def rebuild(cls, mem_id):
        # Called on client side
        pid, fd = mem_id
        local_fd = os.open(f'/proc/{pid}/fd/{fd}', os.O_RDWR)
        return cls(fd=local_fd, mem_id=mem_id)
    
    def get_buffer(self):
        # Called on both side
        self._mmap = mmap.mmap(self._fd, size)
        self._buf = memoryview(self._mmap)
        return self._buf
```


### Integrate with pickling

As we discussed before, we need to modify the pickling process of `UntypedStorage`. Similar to what was implemented in `torch.multiprocessing.reductions`, we define our custom reduce functions for `pickle`:

```python
# Hoarder and borrower is a wrapper to SharedMemory above, contains
# boring stuff like memory arena, etc.
def _reduce_storage(storage):
    # Called by server
    device = storage.device
    storage = storage.cpu()

    # Store content in shared memory
    # The `frag` contains the complete information needed to locate the content.
    frag = hoarder.put(storage)

    return (_rebuild_storage_on_client, (frag, device))

def _rebuild_storage_on_client(frag, device):
    # Called by client
    mv = borrower.borrow(frag)  # Get a memoryview from shared memory
    storage = _make_untyped_storage(mv)  # Zero-copy!
    if device.type == 'cuda':
        return storage.cuda(device.index)
    return storage

class OvermindPickler(dill.Pickler):
    ...

OvermindPickler.register(torch.storage.UntypedStorage, _reduce_storage)
```

Now, simple `OvermindPickler.dumps` and `OvermindPickler.loads` will utilize shared memory to speed up. You can stop reading here if you are already fed up. The rest are details.


## Devils in the Detail

### Why not PyTorch's in-house tensor sharing?

For 'in-house tensor sharing', I mean `torch.multiprocessing.reductions`.

1. At high level, PyTorch's method are designed for 'passing tensor to subprocess', seems the same but with subtle difference.
2. PyTorch uses POSIX shm to share memory, subject to the limit mentioned earlier.
3. For every tensor(or `UntypedStorage`), PyTorch allocates a dedicated POSIX shm object for it, even it contains only 4 bytes. Each object consumes a fd.
4. PyTorch deallocates the POSIX shm once they are unpickled, makes it unsuitable for our needs. We need to deserialize same pickle stream multiple times.
5. There are a lot of CUDA related sharing logic, which are pure noise and trouble for our use case.

### Why do you say 'tensor data copied multiple times'?

#### For a typical on-disk `torch.load`:
- The on-disk `torch.save` file is read into memory.
- Get the actual `torch.UntypedStorage` data as `bytes` by Zip file extraction (`torch.save` generates a zip file).
- C++ code will copy the data into its own managed memory in `torch.UntypedStorage` constructor.

#### For a naive `pickle.dumps` and later `pickle.loads`:
- The generated pickle stream internally embeds another pickle stream, `pickle.loads` will copy the inner stream into a new `bytes`.
- `torch.UntypedStorage` data embeds in the inner pickle stream, another copy happens at construction of `torch.UntypedStorage`.
- C++ code will copy the data into its own managed memory in `torch.UntypedStorage` constructor.

### `diffusers` have a dynamic module

Model repos can include Python files that get imported at runtime into a `diffusers_modules` namespace. The client doesn't have these in `sys.path`, breaking unpickling. Fortunately, `diffusers` will write these dynamic Python files on disk, so we can just import the module and call it a day.

```python
def diffusers_dyn_module_workaround():
    from diffusers.utils.constants import HF_MODULES_CACHE
    modpath = Path(HF_MODULES_CACHE) / "diffusers_modules/__init__.py"
    spec = importlib.util.spec_from_file_location("diffusers_modules", modpath)
    sys.modules["diffusers_modules"] = importlib.util.module_from_spec(spec)
```

### Support for `bitsandbytes`

The most annoying thing about supporting `bitsandbytes` is that the quantization process happens on a GPU. Once we initialized CUDA and torch in `overmind` server, there is no easy way to uninitialize it, which can cause problems for real workloads (mainly less usable VRAM). Therefore, we modified our server to spawn a subprocess, load it into shared memory, and terminate. This happens to improve the stability of `overmind` server.

The quantized parameters are special subclasses provided by `bitsandbytes`. They weren't designed with 'picklability' in mind, so we have to do it ourselves.

```python
def _reduce_bnb_param(p):
    dev = p._prev_device
    assert p.quant_state
    return (_rebuild_bnb_param, (type(p), p.data, p.quant_state.as_dict(packed=True), dev))


def _rebuild_bnb_param(typ, data, qs_dict, dev):
    return typ.from_prequantized(data, qs_dict, device=dev)


def bitsandbytes_quirks():
    try:
        import bitsandbytes
    except ImportError as e:
        return

    ForkingPickler.register(bitsandbytes.nn.modules.Params4bit, _reduce_bnb_param)
    ForkingPickler.register(bitsandbytes.nn.modules.Int8Params, _reduce_bnb_param)
```

Quantized models via `bitsandbytes` come with hooks and monkey-patches that don't pickle, we must strip them:

```python
from accelerate.hooks import remove_hook_from_module
remove_hook_from_module(model, True)
model.__dict__.pop('to', None)  # Remove warning monkeypatches
model.__dict__.pop('cuda', None)
```

We have also encountered issues where functions are nested within other functions (rather than being at the top level), which makes them not picklable. We tried to workaround this, but with no luck. We had to switch our pickle from stdlib provided one to `dill` to pickle this. `dill` is much more powerful, but it's a pure Python implementation, which is much slower than the standard library version. Fortunately, this cost will only be paid once when we are loading model first time (only affects pickling, not unpickling).


### Support for `stable-fast`

`stable-fast` generates `torch.compile` results, which can't be pickled. But with `torch.jit.save`, we could save the results as a zip file. This sounds inefficient, but it's better than nothing.

With only `torch.jit.save` it's not sufficient to pickle `stable-fast` results. `stable-fast` uses a 'flatten' process to make the Torch module traceable. When encountering something it doesn't recognize (for example, the `dataclass`'s class), it won't serialize it, but will only keep a reference to the actual class. We have patched the relevant logic to actually store a pickled class within the 'flatten'ed stream.

```python
def stable_fast_quirks():
    ...

    # pickle dataclass type instead of just put it into a container (which will not survive after torch.jit.save)
    def flatten_dataclass(obj):
        from sfast.utils.flat_tensors import flatten_bytes, flatten_dict
        import dataclasses
        d = dict((field.name, getattr(obj, field.name))
                for field in dataclasses.fields(obj))
        import pickle
        pickled = pickle.dumps(obj.__class__)
        return flatten_bytes(pickled) + flatten_dict(d)

    def unflatten_dataclass(tensors, start):
        from sfast.utils.flat_tensors import unflatten_bytes, unflatten_dict
        import pickle
        pickled, start = unflatten_bytes(tensors, start)
        clz = pickle.loads(pickled)
        content, start = unflatten_dict(tensors, start)
        return clz(**content), start

    sfast.utils.flat_tensors.flatten_dataclass = flatten_dataclass
    sfast.utils.flat_tensors.unflatten_dataclass = unflatten_dataclass
```

There are two more tricks here:

1. We repack the ZIP file with `ZIP_STORED`, so we don't have to decompress the ZIP file for every subsequent load.
2. The `torch.jit.load` interface also incurs the memory copy issue, so we wrote a simple wrapper to load it via the Python buffer protocol, just like `UntypedStorage`.

```cpp
void initOvermindHelpers(py::module m) {
    // ...
    m.def("import_ir_module_from_buffer_0copy",
        [](std::shared_ptr<torch::jit::CompilationUnit> cu, py::buffer buffer) {
            auto info = buffer.request();
            imemstream in((char*)info.ptr, info.size);  // No copy!
            return import_ir_module(std::move(cu), in, ...);
        }
    );
}
```

### The `vae=vae` pattern

Our codebase has something like this, it attempts to load a model with a previous loaded model as its argument:

```python
import overmind.api
overmind.api.monkey_patch_all()

import torch
from diffusers.models import AutoencoderKL

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)

vae = AutoencoderKL.from_pretrained(
    "lemon2431/ChineseInkComicStrip_v10",
    subfolder="vae",
    torch_dtype=torch.float16,
)
controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16,
    variant="fp16",
)
controlnet_edge = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_softedge",
    torch_dtype=torch.float16,
    variant="fp16",
)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "lemon2431/ChineseInkComicStrip_v10",
    vae=vae,  # Here !
    controlnet=[controlnet_edge, controlnet_depth],  # and Here!
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipeline.to('cuda')
```

As we mentioned earlier, the function arguments are assumed to be simple, easily picklable objects, but this pattern breaks that assumption. To handle this, we added special logic: each cached result gets an ID attached. If that object is used as an argument in another call, the client replaces it with its ID, and the server can then recover the actual object based on the ID.

The resulting `pipeline` model will contain a reference to `vae`. For simplicity, we just pickle it directly here. However, when moving the actual `UntypedStorage` to shared memory, we deduplicate any repeated data.

We may have used pickle's `persistent_id` mechanism, but I didn't try this route. That's a bit of a shame.

## Benchmarking

And now for the part that everyone loves to see.

We use the VAE pattern script of the last section to do our test.


| Test     | `vae` | `depth` | `edge` | `pipeline` | to('cuda') | Total |
|----------|-------|---------|--------|------------|------------|-------|
| w/o, 1st | 1.18  | 0.98    | 1.41   | 1.65       | 0.91       | 6.16  |
| w/o, 2nd | 1.15  | 0.96    | 0.97   | 1.65       | 0.89       | 5.66  |
| w/o, 3rd | 1.15  | 0.96    | 0.98   | 1.61       | 0.91       | 5.65  |
| w/o, 4th | 1.42  | 1.10    | 1.11   | 1.72       | 0.88       | 6.27  |
| w/o, 5th | 1.28  | 1.08    | 1.10   | 1.72       | 0.92       | 6.13  |
| w/,  1st | 5.44  | 5.17    | 5.41   | 7.29       | 0.86       | 24.20 |
| w/,  2nd | 0.00  | 0.01    | 0.01   | 0.20       | 0.87       | 1.12  |
| w/,  3rd | 0.01  | 0.01    | 0.01   | 0.21       | 0.86       | 1.12  |
| w/,  4th | 0.01  | 0.01    | 0.01   | 0.20       | 0.90       | 1.15  |
| w/,  5th | 0.01  | 0.01    | 0.01   | 0.21       | 0.86       | 1.13  |

As you can see, the initial load with `overmind` takes 24.2 seconds, which is significantly longer compared to loading without it. However, on subsequent loads, only `.to('cuda')` cost is still present.

Adding up the sizes of all the serialized model files, the whole pipeline is estimated to use around 5808 megabytes of memory. A quick benchmark gives a similar result.

```
In [1]: t = torch.ones((5808, 1024, 1024), dtype=torch.uint8)

In [2]: %time a = t.cuda()
CPU times: user 976 ms, sys: 874 μs, total: 977 ms
Wall time: 976 ms

In [3]: %timeit a = t.cuda()
1.01 s ± 56.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Tested on Intel i9-11900K + GeForce RTX 4090.

## Unexpected Side Effects (Positive!)

Our primary motivation for building `overmind` was to enable rapid switching of model weights during inference. While it served its purpose, we discovered several additional advantages along the way.

We deploy multiple instances of our application, one for each GPU. Thus, there will be 8 processes per node. After we deployed `overmind`, the system memory usage was reduced dramatically. We weren't suffering from system memory shortage, but if we had been, this would have been a big win.

Later, we found it to be a great boost to our algorithm and pipeline developers. For each modify-verify loop, we could save 10 to 20 seconds of loading time, this could add up to a huge number. More importantly, the seconds saved could keep developers in the flow. 


## Github

We are open-sourcing it on [Github](https://github.com/taichi-dev/overmind), we'll be happy if it helped.