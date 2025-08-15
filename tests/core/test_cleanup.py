import logging
import pytest
from maize.core import runtime

class DummyRunnable(runtime.Runnable):
    """Minimal runnable for testing cleanup_processes."""
    def __init__(self, name="dummy"):
        self.name = name
        self.signal = None
        self.n_signals = 0
        self.cleaned_up = False
    
    def execute(self):
        pass
    
    def cleanup(self):
        self.cleaned_up = True

        
class DummyProcess:
    """Minimal process-like object for testing cleanup_processes."""
    def __init__(self, name="dummy-proc",alive=False):
        self.name = name
        self.pid = 1234
        self.exitcode = 0
        self._alive=alive
        self.join_called = False
        self.kill_called = False
    
    def join(self, timeout=None):
        self.join_called = True
    
    def is_alive(self):
        return self._alive
    
    def kill(self):
        self.kill_called = True
        self._alive=False
    
def test_cleanup_equal_lengths(caplog):
    """Case 1, len(procs) == len(items), everything cleaned up in __exit__ function."""
    caplog.set_level(logging.DEBUG)
    items = [DummyRunnable(f"item{i}") for i in range(2)]
    procs = [DummyProcess(f"proc{i}") for i in range(2)]

    runtime.cleanup_processes(items, procs, wait_time = 1.0)

    assert all(i.cleaned_up for i in items)
    assert procs == []
    assert "Joined runnable" in caplog.text

def test_cleanup_procs_longer_than_items(caplog):
    """Case 2, len(procs) > len(items) should raise RuntimeError."""
    items = [DummyRunnable("item1")]
    procs = [DummyProcess("proc1"), DummyProcess("proc2")]
    with pytest.raises(RuntimeError):
        runtime.cleanup_processes(items, procs, wait_time = 1.0)
    assert procs == []
    assert "Killed" in caplog.text

def test_cleanup_items_longer_than_procs(caplog):
    """Case 3, len(procs) < len(items) clean matching, skip extra items."""
    caplog.set_level(logging.DEBUG)
    items = [DummyRunnable("item1"), DummyRunnable("item2"), DummyRunnable("item3")]
    procs = [DummyProcess("proc1",alive=False)]

    runtime.cleanup_processes(items,procs, wait_time = 1.0)
    assert items[0].cleaned_up
    assert not items[1].cleaned_up or not items[2].cleaned_up
    assert "Joined runnable" in caplog.text

def test_cleanup_via_atexit(caplog):
    """Case 4, simulate atexit backup call after __exit__ cleanup."""
    caplog.set_level(logging.DEBUG)

    items = [DummyRunnable("item1"), DummyRunnable("item2")]
    procs = [DummyProcess("proc1", alive=True)]

    runtime.cleanup_processes(items, procs, wait_time = 1.0)

    #alive proc1 should have been killed
    assert procs == []
    assert "Killed" in caplog.text