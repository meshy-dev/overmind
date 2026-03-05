# Overmind：一种非侵入式的模型加载加速方案，将加载时间从 15s 缩短到 0.2s

## TL;DR

ML 模型加载很慢，即便 Linux page cache 已经热好了也一样。所以我们做了一个库来解决这个问题。
其中有一些有意思的技术细节值得分享，于是写了这篇文章。
另外这个库还带来了一些意想不到的好处，文末会提到。

## 起因

故事要从两年前说起。当时我们上线了第一版低多边形（lowpoly）生成模式。这个模式效果很差——以现在的眼光来看简直不堪入目——但代价却不小：一张专用 GPU 每天只能处理个位数的任务。它用的是微调过的权重，大到能把其他所有模型权重都挤出显存。更糟糕的是，这样的模型大概有 3 个（具体数字记不清了），它们占据了推理基础设施的很大一部分，性价比极低。也不能简单地按需加载——光加载就要 30 秒，比实际推理时间还长。

那时候我们还没有专职的 pipeline 工程师，算法同学只能自己想办法，把模型权重在 CPU 和 GPU 之间来回倒腾。搞了几天，代码里到处都是 `this.to('cpu')` 和 `that.to('cuda')`。这种方式短期内能用，但时不时就会打断算法同学的开发节奏。能不能让这些事情自动发生？毕竟是 Python，Python 里什么事都能自动发生。

## 怎样算"自动"？

我们站在算法开发者的角度来想这个问题。诉求很明确：除非万不得已，我不想关心核心算法之外的运行时性能问题，最好完全不知道模型在换进换出。

当然，完全做到这一点不现实，但我们可以尽量减少对算法代码的侵入。这让我想到了 `gevent` 库的 monkey-patching 机制——它（主要）替换了 `socket` 库，换成 `gevent.socket`，在 IO 阻塞时自动切换到其他 greenlet，有点像 goroutine（实际上 `gevent` 比 Go 语言还早！）。

当时我们统一使用 HuggingFace 的库（`transformers`、`diffusers`）来加载模型，所以目标很明确：只需要加一行 monkey-patch 调用，其余代码不用动，`XXXPipeline.from_pretrained(...)` 就能快起来。

## 一些事实、显而易见的决策和前提假设

**Overmind 本质上是一个缓存库，它把模型加载的结果缓存到系统内存中，之后再快速重建。**

monkey-patching 的具体实现不展开讲了，没什么特别有趣的。只需要知道它会把所有 `XXXPipeline.from_pretrained(...)` 调用重定向到 `overmind.api.load(XXXPipeline.from_pretrained, ...)`。

序列化方面我们用的是 `pickle`……也没得选，`torch.save` 本身就用 `pickle`，不用反而奇怪。

我们采用了 C/S 架构，这样进程退出后缓存不会失效，同时子进程也能直接复用已有的缓存。

我们假设 `XXXPipeline.from_pretrained` 的参数都是简单的可哈希对象（`str` 之类），或者是其他由 `overmind` 加载的模型（后面会解释）。

你可能已经猜到了，`overmind` 这个名字借自《星际争霸》。

## 快速重建！

不能简单地把 `pickle.loads` 的结果存在内存里就完事。毕竟在 page cache 已经热好的情况下，Linux 已经帮我们缓存了磁盘上的模型文件，加载时间仍然要十几秒。

瓶颈在于内存拷贝。在 Python 中，创建几百万个对象也就百毫秒级别。但拷贝 10GiB 的内存，光这一步就要半秒。我们必须尽可能避免内存拷贝。

好在大块内存基本都是 Torch 张量，我们只需要关注它们，其余的可以忽略。

在研究张量共享机制时，我从 reduction 代码中了解到了 Torch 张量的内部结构：

```python
# 摘自 torch.multiprocessing.reductions，省略了大部分代码
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

很简单：一个张量就是它的类型、元数据和底层存储。这里的 `storage` 类型是 `TypedStorage`，但 `TypedStorage` 只是 `UntypedStorage` 的简单封装。`UntypedStorage` 才是真正持有张量数据的类。

现在问题变得更具体了：如何避免拷贝 `UntypedStorage`？能不能自己管理张量内存，然后让 `UntypedStorage` 直接指向我们管理的内存？

答案是可以！

翻一下 `UntypedStorage` 构造相关的 C++ 代码，很容易就能找到这样的代码段：

```cpp
// 摘自 torch/csrc/Storage.cpp
static PyObject* THPStorage_get(THPStorage* self, PyObject* index) {
    // ...省略无关代码...

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

我们不仅可以传入指针，`at::DataPtr` 还支持自定义析构函数，这让生命周期管理变得简单很多。

在 Python 这边，指向一段内存的方式是 `memoryview` 对象，它实现了 buffer protocol。很多东西都能产出 `memoryview`，其中 `bytes` 和 `mmap` 是最主要的两个，也是我们关心的。

最终方案就是：写一个函数，接收 `memoryview` 对象，零拷贝地构造出 `UntypedStorage`。有了这个能力，张量的实际数据就不需要放在 pickle 流里了，大幅减少了需要拷贝的数据量。

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

这就是 `overmind` 的核心构建模块。


## 共享张量！


> **注意：** PyTorch 自带张量共享机制，但不适合我们的场景。稍后会解释原因。


### 首先，在客户端和服务端之间共享内存

看到"共享"和"内存"放在一起，大家都会条件反射地想到 `shmget` 那一套。它确实是"设计"来做内存共享的，对吧？但它有两个致命缺陷：

- POSIX 共享内存是稀缺资源，可用量取决于系统管理员的配置。一个极端但普遍的例子是 Docker 容器，默认只有 64MiB 的 POSIX shm 可用。
- POSIX 共享内存的生命周期比进程长，必须自行管理。如果管理进程被强杀，或者处理不当，shm 对象可能会永远残留在系统中。

仔细翻翻，Linux 里有很多有意思的系统调用。`memfd_create` 就是我们想要的：它返回一个 fd，指向一块匿名内存。你可以对它做各种文件操作：read、write，当然也包括 mmap。只要能共享 fd，就能共享内存。

共享 fd 有一种"标准"但相当晦涩的方式：`sendmsg` 配合 `SCM_RIGHTS`。虽然可以借助第三方库隐藏 `sendmsg` 那些吓人的细节，但仍然需要在服务端和客户端之间做协调。我们用了一个取巧的方法：在客户端直接打开 `/proc/{pidof(server)}/fd/{memfd}`，同时服务端永远不关闭这个 fd。唯一需要传递的信息就是一个 `(pid, fd)` 元组。在我们的场景下完美可用。

以上内容浓缩成代码就是这几行：

```python
class SharedMemory:
    @classmethod
    def create(cls, shift):
        # 在服务端调用
        libc = ctypes.CDLL(None)
        name = _make_filename(shift).encode('utf-8')
        fd = libc.memfd_create(name, os.O_RDWR)
        os.ftruncate(fd, 1 << shift)
        mem_id = (os.getpid(), fd)
        return cls(fd=fd, mem_id=mem_id)

    @classmethod
    def rebuild(cls, mem_id):
        # 在客户端调用
        pid, fd = mem_id
        local_fd = os.open(f'/proc/{pid}/fd/{fd}', os.O_RDWR)
        return cls(fd=local_fd, mem_id=mem_id)

    def get_buffer(self):
        # 两端都会调用
        self._mmap = mmap.mmap(self._fd, size)
        self._buf = memoryview(self._mmap)
        return self._buf
```


### 与 pickle 集成

前面提到，我们需要修改 `UntypedStorage` 的序列化过程。参考 `torch.multiprocessing.reductions` 的做法，我们为 `pickle` 定义了自定义的 reduce 函数：

```python
# Hoarder 和 borrower 是 SharedMemory 的封装，包含内存池等
# 不太有趣的实现细节。
def _reduce_storage(storage):
    # 服务端调用
    device = storage.device
    storage = storage.cpu()

    # 将内容存入共享内存
    # `frag` 包含定位内容所需的全部信息
    frag = hoarder.put(storage)

    return (_rebuild_storage_on_client, (frag, device))

def _rebuild_storage_on_client(frag, device):
    # 客户端调用
    mv = borrower.borrow(frag)  # 从共享内存获取 memoryview
    storage = _make_untyped_storage(mv)  # 零拷贝！
    if device.type == 'cuda':
        return storage.cuda(device.index)
    return storage

class OvermindPickler(dill.Pickler):
    ...

OvermindPickler.register(torch.storage.UntypedStorage, _reduce_storage)
```

现在，简单的 `OvermindPickler.dumps` 和 `OvermindPickler.loads` 就能利用共享内存来加速了。如果你已经看够了，可以到此为止。下面是一些细节。


## 亿点细节

### 为什么不用 PyTorch 自带的张量共享？

这里说的"自带张量共享"是指 `torch.multiprocessing.reductions`。

1. 从设计意图来看，PyTorch 的方案是为"传递张量给子进程"设计的，看起来类似但存在微妙的差异。
2. PyTorch 使用 POSIX shm 来共享内存，受限于前面提到的配额问题。
3. 每个张量（或 `UntypedStorage`）都会分配一个独立的 POSIX shm 对象，即使只有 4 字节。每个对象都会占用一个 fd。
4. PyTorch 在反序列化完成后就会释放 POSIX shm，不适合我们的需求——我们需要多次反序列化同一份 pickle 数据流。
5. 里面有大量 CUDA 相关的共享逻辑，对我们来说纯属噪音和麻烦。

### 为什么说"张量数据被多次拷贝"？

#### 典型的 `torch.load` 从磁盘加载：
- 磁盘上的 `torch.save` 文件被读入内存。
- 通过 Zip 解压提取出 `torch.UntypedStorage` 的实际数据（`torch.save` 生成的是 zip 文件），解压为 `bytes`。
- C++ 层在 `torch.UntypedStorage` 构造函数中再把数据拷贝到自己管理的内存里。

#### 简单粗暴的 `pickle.dumps` 再 `pickle.loads`：
- 生成的 pickle 流内部嵌套了另一个 pickle 流，`pickle.loads` 会把内层流拷贝成新的 `bytes`。
- `torch.UntypedStorage` 的数据嵌在内层 pickle 流中，构造 `torch.UntypedStorage` 时又发生一次拷贝。
- C++ 层在 `torch.UntypedStorage` 构造函数中再拷贝一次到自己管理的内存里。

### `diffusers` 的动态模块

模型仓库可能包含 Python 文件，在运行时被导入到 `diffusers_modules` 命名空间。客户端的 `sys.path` 里没有这些文件，反序列化就会失败。好在 `diffusers` 会把这些动态 Python 文件写到磁盘上，所以直接导入就行了。

```python
def diffusers_dyn_module_workaround():
    from diffusers.utils.constants import HF_MODULES_CACHE
    modpath = Path(HF_MODULES_CACHE) / "diffusers_modules/__init__.py"
    spec = importlib.util.spec_from_file_location("diffusers_modules", modpath)
    sys.modules["diffusers_modules"] = importlib.util.module_from_spec(spec)
```

### 对 `bitsandbytes` 的支持

支持 `bitsandbytes` 最让人头疼的是量化过程需要用 GPU。一旦在 `overmind` 服务端初始化了 CUDA 和 torch，就很难反初始化，这会影响实际工作负载（主要是显存可用量减少）。因此我们改成让服务端 fork 一个子进程来完成加载，加载到共享内存后子进程就退出。这顺带还提升了 `overmind` 服务端的稳定性。

量化后的参数是 `bitsandbytes` 提供的特殊子类，设计时没考虑过序列化，所以只能我们自己来。

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

通过 `bitsandbytes` 量化的模型会带有 hook 和 monkey-patch，这些东西没法序列化，必须先去掉：

```python
from accelerate.hooks import remove_hook_from_module
remove_hook_from_module(model, True)
model.__dict__.pop('to', None)  # 移除告警用的 monkeypatch
model.__dict__.pop('cuda', None)
```

我们还遇到过函数定义嵌套在其他函数内部（而不是在模块顶层）的情况，这类函数无法被 pickle 序列化。尝试绕过未果，最终把 pickle 从标准库换成了 `dill`。`dill` 功能强大得多，但它是纯 Python 实现，比标准库慢不少。好在这个代价只在首次加载模型时需要付出（只影响序列化，不影响反序列化）。


### 对 `stable-fast` 的支持

`stable-fast` 会生成 `torch.compile` 的结果，这些结果不能直接 pickle。但通过 `torch.jit.save` 可以把它保存为 zip 文件。虽然听起来效率不高，但总比没有好。

仅靠 `torch.jit.save` 还不够。`stable-fast` 用了一套"展平"（flatten）流程来让 Torch 模块可追踪。遇到它不认识的东西（比如 `dataclass` 的类），它不会序列化，只保留一个引用。我们打了个补丁，让展平后的数据流里真正存入 pickle 后的类信息。

```python
def stable_fast_quirks():
    ...

    # 将 dataclass 类型 pickle 存储，而不是仅仅放入容器中（那样在 torch.jit.save 后就丢了）
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

这里还有两个小技巧：

1. 我们用 `ZIP_STORED` 重新打包 ZIP 文件，这样后续加载时就不用每次都解压了。
2. `torch.jit.load` 接口同样存在内存拷贝问题，所以我们写了个简单的封装，通过 Python buffer protocol 来加载，和 `UntypedStorage` 的处理方式一样。

```cpp
void initOvermindHelpers(py::module m) {
    // ...
    m.def("import_ir_module_from_buffer_0copy",
        [](std::shared_ptr<torch::jit::CompilationUnit> cu, py::buffer buffer) {
            auto info = buffer.request();
            imemstream in((char*)info.ptr, info.size);  // 零拷贝！
            return import_ir_module(std::move(cu), in, ...);
        }
    );
}
```

### `vae=vae` 模式

我们的代码库里有类似这样的用法——加载一个模型时，把之前加载好的模型作为参数传入：

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
    vae=vae,  # 注意这里！
    controlnet=[controlnet_edge, controlnet_depth],  # 还有这里！
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipeline.to('cuda')
```

前面提到，我们假设函数参数都是简单的可序列化对象，但这种模式打破了这个假设。为了应对，我们加了一段特殊逻辑：每个缓存结果会附带一个 ID。当这个对象作为参数传给另一个调用时，客户端会用 ID 替换它，服务端再根据 ID 恢复出实际对象。

最终的 `pipeline` 模型会包含对 `vae` 的引用。为了简单起见，我们直接序列化。不过在将 `UntypedStorage` 存入共享内存时，我们会对重复数据做去重。

本来可以用 pickle 的 `persistent_id` 机制来做，但我没有尝试这条路。有点小遗憾。

## 性能测试

接下来是大家最爱看的部分。

我们用上一节的 VAE 模式脚本来做测试。


| 测试项        | `vae` | `depth` | `edge` | `pipeline` | to('cuda') | 总计  |
|---------------|-------|---------|--------|------------|------------|-------|
| 无缓存, 第1次 | 1.18  | 0.98    | 1.41   | 1.65       | 0.91       | 6.16  |
| 无缓存, 第2次 | 1.15  | 0.96    | 0.97   | 1.65       | 0.89       | 5.66  |
| 无缓存, 第3次 | 1.15  | 0.96    | 0.98   | 1.61       | 0.91       | 5.65  |
| 无缓存, 第4次 | 1.42  | 1.10    | 1.11   | 1.72       | 0.88       | 6.27  |
| 无缓存, 第5次 | 1.28  | 1.08    | 1.10   | 1.72       | 0.92       | 6.13  |
| 有缓存, 第1次 | 5.44  | 5.17    | 5.41   | 7.29       | 0.86       | 24.20 |
| 有缓存, 第2次 | 0.00  | 0.01    | 0.01   | 0.20       | 0.87       | 1.12  |
| 有缓存, 第3次 | 0.01  | 0.01    | 0.01   | 0.21       | 0.86       | 1.12  |
| 有缓存, 第4次 | 0.01  | 0.01    | 0.01   | 0.20       | 0.90       | 1.15  |
| 有缓存, 第5次 | 0.01  | 0.01    | 0.01   | 0.21       | 0.86       | 1.13  |

可以看到，使用 `overmind` 的首次加载需要 24.2 秒，比不用的时候慢很多。但后续加载中，唯一的耗时就只剩 `.to('cuda')` 了。

把所有序列化模型文件的大小加起来，整个 pipeline 大约占用 5808 MB 内存。简单跑个 benchmark 也能得到相近的结果。

```
In [1]: t = torch.ones((5808, 1024, 1024), dtype=torch.uint8)

In [2]: %time a = t.cuda()
CPU times: user 976 ms, sys: 874 μs, total: 977 ms
Wall time: 976 ms

In [3]: %timeit a = t.cuda()
1.01 s ± 56.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

测试环境：Intel i9-11900K + GeForce RTX 4090。

## 意外收获

我们做 `overmind` 的初衷是在推理过程中快速切换模型权重。虽然这个目标达成了，但一路走来还发现了不少额外的好处。

我们的应用按每 GPU 一个实例部署，所以每个节点会有 8 个进程。部署 `overmind` 之后，系统内存使用量大幅下降。我们当时其实并不缺系统内存，但如果缺的话，这个收益就非常可观了。

后来我们发现，它对算法和 pipeline 开发者的效率提升非常明显。每次修改-验证的循环都能省下 10 到 20 秒的加载时间，累积起来相当可观。更重要的是，省下来的这几秒能让开发者保持在心流状态。


## Github

我们已经在 [Github](https://github.com/meshy-dev/overmind) 上开源了这个项目，希望能帮到你。
