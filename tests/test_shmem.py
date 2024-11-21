def test_hoarder_and_stealer():
    from overmind.shmem import Filler, Borrower
    hoarder = Filler()
    stealer = Borrower()
    frag = hoarder.put(b'Hello World!!!')
    assert bytes(stealer.borrow(frag)) == b'Hello World!!!'
