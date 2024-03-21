import sys

if sys.platform == 'win32':
    import ctypes
    from ctypes.wintypes import BOOL, HANDLE, DWORD, LPCWSTR

    # Load the necessary functions from kernel32.dll
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    # Define necessary constants
    CREATE_MUTEX_INITIAL_OWNER = 0x00000001
    INFINITE = 0xFFFFFFFF
    WAIT_OBJECT_0 = 0x00000000
    WAIT_ABANDONED = 0x00000080
    WAIT_TIMEOUT = 0x00000102
    WAIT_FAILED = 0xFFFFFFFF

    class Win32Mutex:
        def __init__(self, name: str):
            self.name = name
            self.mutex = kernel32.CreateMutexW(None, False, name)
            if not self.mutex:
                raise ctypes.WinError(ctypes.get_last_error())

        def acquire(self) -> bool:
            result = kernel32.WaitForSingleObject(self.mutex, 0)  # 0 timeout for non-blocking
            if result in (WAIT_OBJECT_0, WAIT_ABANDONED):
                return True  # Mutex was successfully acquired
            elif result == WAIT_TIMEOUT:
                return False  # Mutex could not be acquired
            else:
                raise ctypes.WinError(ctypes.get_last_error())

        def release(self):
            if not kernel32.ReleaseMutex(self.mutex):
                raise ctypes.WinError(ctypes.get_last_error())

        def __del__(self):
            if self.mutex:
                kernel32.CloseHandle(self.mutex)
