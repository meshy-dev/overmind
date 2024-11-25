def test_hoarder_and_stealer():
    from overmind.shmem import Hoarder, Borrower
    hoarder = Hoarder()
    borrower = Borrower()
    frag = hoarder.put(b'Hello World!!!')
    assert bytes(borrower.borrow(frag)) == b'Hello World!!!'
