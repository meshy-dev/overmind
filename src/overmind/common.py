class OvermindObjectRef(str):
    __slots__ = ()

    def __repr__(self):
        return f'OvermindObjectRef({super().__repr__()})'
