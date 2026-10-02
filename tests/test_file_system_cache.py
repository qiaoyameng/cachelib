import hashlib
import os
from time import sleep
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from clear import ClearTests
from common import CommonTests
from has import HasTests
from serializer import SerializerTests

from cachelib import FileSystemCache
from cachelib.serializers import BaseSerializer


class SillySerializer(BaseSerializer):
    """A pointless serializer only for testing"""

    def dump(self, value, fs):
        fs.write(f"{repr(value)}{os.linesep}".encode())

    def load(self, fs):
        try:
            loaded = eval(fs.readline().decode())
        # When all file content has been read eval will
        # turn the EOFError into SyntaxError which is not
        # handled by cachelib
        except SyntaxError as e:
            raise EOFError from e
        return loaded


class CustomSerializerCache(FileSystemCache):
    """Our custom cache client with non-default serializer"""

    # overwrite serializer
    serializer = SillySerializer()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CustomHashingMethodCache(FileSystemCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hash_method=hashlib.sha256, **kwargs)


@pytest.fixture(
    autouse=True,
    params=[
        FileSystemCache,
        CustomSerializerCache,
        CustomHashingMethodCache,
    ],
)
def cache_factory(request, tmpdir):
    def _factory(self, *args, **kwargs):
        client = request.param(tmpdir, *args, **kwargs)
        return client

    request.cls.cache_factory = _factory


class TestFileSystemCache(CommonTests, ClearTests, HasTests, SerializerTests):
    # override parent sample since these must implement buffer interface
    sample_pairs = {
        "bacon": "eggs",
        "sausage": "spam",
        "brandy": "lobster",
        "truffle": "wine",
        "sauce": "truffle pate",
        "cravettes": "mournay sauce",
    }

    def test_EOFError(self, caplog):
        cache = self.cache_factory(threshold=1)
        assert cache.set_many(self.sample_pairs)
        file_names = [cache._get_filename(k) for k in self.sample_pairs.keys()]
        # truncate files to erase content
        for fpath in file_names:
            open(fpath, "w").close()
        assert cache.set("test", "EOFError")
        assert "Exception raised" in caplog.text

    def test_threshold(self):
        threshold = len(self.sample_pairs) // 2
        cache = self.cache_factory(threshold=threshold)
        assert cache.set_many(self.sample_pairs)
        assert cache._file_count == 4
        # due to autouse=True a single tmpdir is used
        # for each test so we need to clear it
        assert cache.clear()
        cache = self.cache_factory(threshold=0)
        assert cache.set_many(self.sample_pairs)
        assert not cache._file_count

    def test_file_counting(self):
        cache = self.cache_factory()
        assert cache.set_many(self.sample_pairs)
        assert cache._file_count == len(self.sample_pairs)
        assert cache.clear()
        assert cache._file_count == 0

    def test_file_counting_on_override(self):
        cache = self.cache_factory()
        assert cache.set_many(self.sample_pairs)
        assert cache._file_count == len(self.sample_pairs)
        assert cache.set_many(self.sample_pairs)
        # count should remain the same
        assert cache._file_count == len(self.sample_pairs)

    def test_prune_old_entries(self):
        threshold = 2 * len(self.sample_pairs) - 1
        cache = self.cache_factory(threshold=threshold)
        for k, v in self.sample_pairs.items():
            assert cache.set(f"{k}-t1", v, timeout=1)
            assert cache.set(f"{k}-t10", v, timeout=10)
        sleep(3)
        for k, v in self.sample_pairs.items():
            assert cache.set(k, v)
            assert cache.has(f"{k}-t10")
            assert not cache.has(f"{k}-t1")

    def test_run_safely_succeeds_on_first_try(self):
        """_run_safely returns the function's result when no error is raised."""
        cache = self.cache_factory()
        result = cache._run_safely(lambda: 42)
        assert result == 42

    def test_run_safely_retries_on_permission_error(self):
        """
        _run_safely retries when PermissionError is raised
        (e.g. Windows NTFS race).
        """
        cache = self.cache_factory()
        flaky = Mock(
            side_effect=[PermissionError("locked"), PermissionError("locked"), "ok"]
        )

        with patch("cachelib.file.sleep") as mock_sleep:
            result = cache._run_safely(flaky)

        assert result == "ok"
        assert flaky.call_count == 3
        assert mock_sleep.call_count == 2

    def test_run_safely_exhausts_retries_and_returns_none(self):
        """
        _run_safely returns None when PermissionError persists
        beyond max_sleep_time.
        """
        cache = self.cache_factory()

        def always_fails():
            raise PermissionError("locked")

        with patch("cachelib.file.sleep"):
            result = cache._run_safely(always_fails)

        assert result is None

    def test_run_safely_does_not_retry_other_errors(self):
        """_run_safely does not catch non-PermissionError exceptions."""
        cache = self.cache_factory()

        def raises_file_not_found():
            raise FileNotFoundError("missing")

        with pytest.raises(FileNotFoundError):
            cache._run_safely(raises_file_not_found)

    def test_run_safely_exponential_backoff(self):
        """_run_safely doubles the wait_step on each retry."""
        cache = self.cache_factory()
        flaky = Mock(
            side_effect=[
                PermissionError("locked"),
                PermissionError("locked"),
                PermissionError("locked"),
                "ok",
            ]
        )

        with patch("cachelib.file.sleep") as mock_sleep:
            cache._run_safely(flaky)

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # Each wait should double the previous
        assert sleep_calls[1] == sleep_calls[0] * 2
        assert sleep_calls[2] == sleep_calls[1] * 2

    def test_mgmt_element_set_not_counted(self, tmpdir):
        """set with mgmt_element=True should not affect _file_count."""
        cache = FileSystemCache(tmpdir)
        before = cache._file_count
        cache.set("__internal_key", 42, mgmt_element=True)
        assert cache._file_count == before

    def test_mgmt_element_delete_not_counted(self, tmpdir):
        """delete with mgmt_element=True should not affect _file_count."""
        cache = FileSystemCache(tmpdir)
        cache.set("__internal_key", 42, mgmt_element=True)
        before = cache._file_count
        cache.delete("__internal_key", mgmt_element=True)
        assert cache._file_count == before

    def test_has_on_expired_key_returns_false(self, tmpdir):
        from time import sleep

        cache = FileSystemCache(tmpdir)
        cache.set("expiring", "value", timeout=1)
        sleep(2)
        assert cache.has("expiring") is False

    def test_max_bytes_accounting(self, tmpdir):
        """Interleaved writes of different sizes and an injected replace
        failure: on-disk totals, readable content and the cache reported
        usage must all agree."""
        max_bytes = 4096
        cache = FileSystemCache(str(tmpdir), threshold=0, max_bytes=max_bytes)

        def disk_usage():
            return sum(os.path.getsize(f) for f in cache._list_dir())

        def leftover_tmp_files():
            return [
                f
                for f in os.listdir(str(tmpdir))
                if f.endswith(cache._fs_transaction_suffix)
            ]

        # interleave small and large values
        assert cache.set("small-1", "s" * 32)
        assert cache.set("large-1", "L" * 1024)
        assert cache.set("small-2", "s" * 32)
        assert cache.set("large-2", "L" * 1024)
        assert cache._file_bytes > 0
        assert cache._file_bytes == disk_usage()

        # replacing a key refunds the old version before charging the new
        old_size = os.path.getsize(cache._get_filename("large-1"))
        before = cache._file_bytes
        assert cache.set("large-1", "L" * 256)
        new_size = os.path.getsize(cache._get_filename("large-1"))
        assert cache._file_bytes == before - old_size + new_size
        assert cache._file_bytes == disk_usage()

        # inject a replace failure: the old value survives, the
        # uncommitted temp file is removed and the counters don't move
        with patch("cachelib.file.os.replace", side_effect=OSError("boom")):
            assert not cache.set("large-2", "X" * 2048)
        assert cache.get("large-2") == "L" * 1024
        assert cache._file_bytes == disk_usage()
        assert not leftover_tmp_files()

        # every stored value is still readable
        assert cache.get("small-1") == "s" * 32
        assert cache.get("small-2") == "s" * 32
        assert cache.get("large-1") == "L" * 256

        # forcing the budget evicts older entries; reported usage, disk
        # usage and readable content stay consistent
        assert cache.set("huge", "H" * 3000)
        assert cache._file_bytes <= max_bytes
        assert cache._file_bytes == disk_usage()
        assert cache.get("huge") == "H" * 3000
        assert not leftover_tmp_files()

    def test_max_bytes_rejects_oversized_entry(self, tmpdir):
        """A single entry larger than the budget is rejected before the
        previous value is destroyed."""
        cache = FileSystemCache(str(tmpdir), threshold=0, max_bytes=1024)
        assert cache.set("key", "small value")
        before = cache._file_bytes
        assert not cache.set("key", "x" * 4096)
        assert cache.get("key") == "small value"
        assert cache._file_bytes == before
        # a fresh oversized key is rejected as well
        assert not cache.set("other", "x" * 4096)
        assert cache.get("other") is None

    def test_max_bytes_eviction_order(self, tmpdir):
        """Expired entries are freed first, then the entries with the
        oldest expiration timestamp."""
        path = str(tmpdir)
        setup = FileSystemCache(path, threshold=0)
        assert setup.set("expired", "e" * 700, timeout=1)
        assert setup.set("oldest", "o" * 700, timeout=100)
        assert setup.set("newest", "n" * 700, timeout=200)
        sleep(2)
        entry_size = os.path.getsize(setup._get_filename("newest"))
        # room for exactly two entries once the budget applies
        cache = FileSystemCache(path, threshold=0, max_bytes=2 * entry_size + 8)
        assert cache._file_bytes == 3 * entry_size
        assert cache.set("trigger", "t" * 700, timeout=300)
        # the expired entry was freed first, then the oldest expiry
        assert not os.path.exists(cache._get_filename("expired"))
        assert not os.path.exists(cache._get_filename("oldest"))
        assert cache.get("newest") == "n" * 700
        assert cache.get("trigger") == "t" * 700
        assert cache._file_bytes <= cache._max_bytes
        assert cache._file_bytes == sum(os.path.getsize(f) for f in cache._list_dir())

    def test_max_bytes_and_threshold_together(self, tmpdir):
        """The entry count and the byte budget are enforced together and
        the stored counters always match the real files on disk."""
        cache = FileSystemCache(str(tmpdir), threshold=100, max_bytes=1024)
        for i in range(8):
            assert cache.set(f"key-{i}", "v" * 256, timeout=100 + i)
        assert cache._file_count <= 100
        assert cache._file_bytes <= 1024
        assert cache._file_count == len(list(cache._list_dir()))
        assert cache._file_bytes == sum(os.path.getsize(f) for f in cache._list_dir())

    def test_serialization_failure_keeps_accounting(self, tmpdir):
        """A serializer failure must not leak temp files or skew the
        stored counters."""

        class ExplodingSerializer(BaseSerializer):
            def dump(self, value, fs):
                raise ValueError("cannot serialize")

        cache = FileSystemCache(str(tmpdir), threshold=10, max_bytes=4096)
        assert cache.set("good", "g" * 100)
        count_before = cache._file_count
        bytes_before = cache._file_bytes
        cache.serializer = ExplodingSerializer()
        with pytest.raises(ValueError, match="cannot serialize"):
            cache.set("bad", "b" * 100)
        assert cache._file_count == count_before
        assert cache._file_bytes == bytes_before
        assert not [
            f
            for f in os.listdir(str(tmpdir))
            if f.endswith(cache._fs_transaction_suffix)
        ]
        assert cache.get("good") == "g" * 100
