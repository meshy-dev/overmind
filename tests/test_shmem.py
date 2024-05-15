def test_hoarder_and_stealer():
    from overmind.shmem import Hoarder, Borrower
    hoarder = Hoarder()
    stealer = Borrower()
    frag = hoarder.put(b'Hello World!!!')
    assert bytes(stealer.borrow(frag)) == b'Hello World!!!'
