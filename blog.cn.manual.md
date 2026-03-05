TL;DR

即使是有预热的 Linux pagecache，模型的加载仍然很慢，所以我们做了一个库，让它变得更快一点。
这里面有很多有趣的技术细节，所以我们写了这篇博客来分享一下。
在文章的结尾，我们还讨论了一些预期外的影响（正面的!）

为什么有这个项目？

事儿要从两年前说起。我们当时刚发布了第一代的 lowpoly 生成模型。它生成的结果从今天的视角来看是非常糟糕的，但是我们花了很大的成本去运行它 —— 一个独占的 GPU， 每天只处理几个零星的任务。这是一个微调过的模型，把它装载到 GPU 上，其他的模型也没有办法在这个 GPU 上运行了。这样的模型我们有 3 个多（具体的数字忘了），这把我们的集群利用率拖到了一个非常低的数值。我们还不能简单地在每次用到相应模型时，再把它装载进 GPU 里。这个过程要花 30 秒，比我们实际处理一个任务的时间都要长。

在当时，我们并没有专门的 AI Infra 工程师，我们的算法工程师就用他们自己的方式在内存中切换模型。几天以后，我们的代码仓库里面到处都是 this.2 CPU 和 that.2 CUDA。这种方式的上限也就到这里了，而且它因为算法工程师需要处理这种细节，会打断他们的心流。这些能不能变得完全透明呢？毕竟是 Python，在 Python 里面让东西变得完全透明，这还是能做到的。

怎么定义“完全透明”？
从算法工程师的视角来看的话，其实挺清晰的：除非我没有办法了，我不想关心核心算法之外的烂糟事情导致的运行时性能问题，我完全不想关心模型 weight 的 swap in 和 swap out。
当然，想一点代码都不改也是不可能的，但我们要试着把对算法代码的侵入性降到最低。这让我想起了 Python 里的 gevent 库。它的 monkey-patching 能力能对 socket 库打补丁，把它自己的 gevent.socket 替换掉标准库中的 socket class。gevent.socket 能在 IO 阻塞的时候切换到其他的协程上去，跟 goroutine 基本上一样（实际上，gevent 比 GoLang 还要老一点！）。

由于我们只用到了 huggingface 的库来加载模型(Transformers 和 Diffusers)，我们的目标就有了：我们在算法的代码上加一个 monkey patch，然后让剩余所有代码中的 XXX_PIPELINE.from_pretrained 自动加速。

事实、假设和一些明显的决策

Overmind 是一个缓存库，它在内存中缓存了模型加载调用的结果，然后可以在后续的调用中快速地重建它。

我们跳过 monkey-patching 是怎么实现的，这不是一个有意思的细节。这里的最终效果就是 MonkeyPatch 会把 XXX_PIPELINE.from_pretrained 的调用，全部指向到 Overmind.api.load(XXX_PIPELINE.from_pretrained).
我们用 pickle 来序列化我们的缓存结果。我们也没得选，torch.save 它自己就用 pickle，我们如果不用的话就很奇怪。
我们会采用客户端/服务端的架构，因为我们不想让缓存随着进程的结束而失效。同时子进程也同样可以用到这些缓存。
我们假设 XXX_PIPELINE.from_pretrained 调用的参数都是一些简单的 hashable 的东西（比如说 string），还有其他的被 Overmind 加载的模型(后面再解释）。
你可能也猜到了，Overmind 这个名字是从《星际争霸》里借用过来的。


快速重建!

我们不能简单地 pickle.loads 在内存中的结果,毕竟在已经预热的情况下，Linux 的 page cache 已经缓存了磁盘上的模型，但我们仍然可以观察到 10 秒钟以上的模型加载时间。
这种低效来源于内存复制. 在 Python 中，即便创建上百万个对象也仅仅花费不到 100 毫秒，但是做一个 10GB 大小的内存复制，将会花费半秒钟的时间。我们必须尽可能地避免内存复制。
所幸绝大部分的内存复制都来源于 Torch tensors，我们可以就处理这一种情况，其他的就不管了。

我在研究 Torch 提供的内建 Tensor 共享机制的时候，学习到了 Torch Tensor 的内部结构:

...code...

挺简单的，一个 Tensor 就是由它的类型、元数据和底层的存储组合起来的。在这里  Storage 是一个 TypeStorage 类型，但实际上 TypeStorage 只是一个 UntypedStorage 的简单包装。 UntypedStorage 是实际上存储 Tensor 数据的类型。

那么，我们能不能尽量避免复制 UntypedStorage？我们能不能自己管理这些 Tensor 的内存，然后通过指针来重建一个 UntypedStorage 呢？可以的!

看一眼 UntypedStorage 被构建的地方的 C++ 代码，我们可以找到下面这个片段：

...code...

我们不仅可以用一个指针，而且 at::DataPtr 这个类也帮我们留了一个析构的接口，我们在管理生命周期的时候会方便得多。
在 Python 这一侧，一个指向内存区域的指针可以被一个 memoryview 对象来表示，它支持 buffer 协议。我们可以从很多东西中拿到一个 memoryview 对象，Bytes 和 mmap 是两个比较明显的例子，而且这两个也是我们主要关心的。

现在我们知道需要干什么了：我们需要一个接受 memoryview 对象，并且返回一个重建好的 UntypedStorage 的函数，过程中不能有内存复制。有了这个能力以后，我们的 Tensor 数据就不是必须在 Pickle 的字节流中，这样我们必须要复制的数据就会少非常多。

...code...

以上就是 Overmind 的最核心的东西.

共享 Tensors!

Note: PyTorch 里已经有一个 Tensor 共享的机制，但是它不适合我们的这个场景，我们之后再讨论它。

首先，在客户端与服务端之间共享内存

当我们说共享内存的时候，我们首先会想到 shmget,毕竟它就是为了共享内存而设计的机制。但是它有两个比较大的缺陷：

POSIX SHM 是一个比较稀缺的资源，你能用多少取决于系统管理员怎么配置系统。有一个极端但是非常普遍的例子：Docker 容器在默认情况下，你只有 64MB 的 POSIX SHM 可以用。
POSIX SHM 的生命周期比你的进程更长，你必须做好管理。如果你的管理进程被强杀，或者是有 bug 没有处理好的话，这些共享内存就会在系统上一直存在着。


如果你仔细地看一下 Linux 的系统调用表，你能发现很多有意思的东西，memfd_create 就是其中的一个: 你可以用它申请一块匿名的内存，它会给你一个 fd 来表示它。你可以对这个 fd 做文件能做的所有事情：读、写，当然还有内存映射(mmap)。如果我们能共享这个 fd，我们就能共享这一块内存。

共享一个 FD 也有一个标准的做法:用 sendmsg 系统调用发送一个 SCM_RIGHTS 的消息. 我们可以通过库来帮助我们处理 sendmsg 系统调用的那些细节，但是我们仍然需要在服务端和客户端之间做好协调。这个地方我们决定用一个 hack: 在客户端，我们直接打开服务端的 /proc/PID_OF_SERVER/fd/MEMFD,同时保证服务端永远不会关掉这些 FD.这样我们的客户端、服务端之间的交流，只是一个简单的 PID、FD 的 tuple. [it works perfectly in our case.]->translate-to-cn

上面说的这些，基本上可以用下面这几行来表示：

...code...

与 Pickle 集成

我们之前讨论过，我们需要改掉 UntypedStorage 的 Pickle 过程. 与 torch.multiprocessing.reductions 相似，我们也定义了自己的 pickle reduce 函数:

...code...

现在 Overmind.pickler.dumps 和 Overmind.pickler.loads 就会利用共享内存来加速。如果你已经烦了，就可以读到这儿为止，剩下的都是些细节。


魔鬼在细节里

GIVING UP!
