def test_make_untyped_storage():
    import torch
    import overmind._C
    a = memoryview(b'Hello World!!!')
    storage = overmind._C._make_untyped_storage(a)
    assert isinstance(storage, torch.UntypedStorage)
    assert bytes(storage) == a

    del a
    del storage

    import gc
    gc.collect()
