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


def test_memcpy_from_untyped_storage():
    import torch
    import overmind._C
    a = bytearray(100)
    t = torch.empty(100, dtype=torch.uint8)
    t.fill_(ord('A'))
    storage = t.untyped_storage()
    overmind._C._memcpy_from_untyped_storage(a, storage)
    assert bytes(a) == b'A' * 100

