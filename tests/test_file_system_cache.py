import hashlib
import os
from time import time
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


def constant_hash(data=b""):
    """A hash method that maps every key to the same digest, forcing
    all keys to collide on a single cache address. The cache management
    key keeps a distinct address, as it would with a real hash."""
    if data == FileSystemCache._fs_count_file.encode("utf-8"):
        return hashlib.sha256(data)
    return hashlib.sha256(b"constant-collision")


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

    def test_collision_safe_interleaved_operations(self, tmpdir):
        """Three keys sharing one hash digest stay fully isolated across
        interleaved add/set/get/has/delete operations."""
        cache = FileSystemCache(
            tmpdir, hash_method=constant_hash, collision_safe=True
        )
        assert cache.set("alpha", 1)
        # a different key on the same address must still be addable
        assert cache.add("beta", 2) is True
        # but an existing key must not be re-added
        assert cache.add("beta", 99) is False
        assert cache.set("gamma", 3)
        assert cache.get("alpha") == 1
        assert cache.get("beta") == 2
        assert cache.get("gamma") == 3
        assert cache.has("alpha")
        assert cache.has("beta")
        assert cache.has("gamma")
        assert cache._file_count == 3
        # overwriting one key leaves the others untouched
        assert cache.set("alpha", 11)
        assert cache.get("alpha") == 11
        assert cache.get("beta") == 2
        assert cache._file_count == 3
        # deleting one key leaves the others untouched
        assert cache.delete("beta") is True
        assert cache.get("beta") is None
        assert cache.has("beta") is False
        assert cache.get("alpha") == 11
        assert cache.get("gamma") == 3
        # deleting an absent key is a successful no-op
        assert cache.delete("beta") is True
        assert cache.delete("missing") is True
        assert cache._file_count == 2
        assert cache.clear()
        assert cache._file_count == 0

    def test_collision_safe_expiry_counts_records_not_files(self, tmpdir):
        """Expiry pruning removes individual records from a shared bucket
        and updates the entry count accordingly."""
        cache = FileSystemCache(
            tmpdir, hash_method=constant_hash, collision_safe=True
        )
        assert cache.set("expiring", "x", timeout=1)
        assert cache.set("stable", "y", timeout=0)
        assert cache._file_count == 2
        # both records live in a single bucket file
        assert len(list(cache._list_dir())) == 1
        sleep(2)
        cache._remove_expired(time())
        assert cache.get("expiring") is None
        assert cache.get("stable") == "y"
        assert cache._file_count == 1

    def test_collision_safe_prune_over_threshold(self, tmpdir):
        cache = FileSystemCache(
            tmpdir, threshold=2, hash_method=constant_hash, collision_safe=True
        )
        assert cache.set("k1", "v1", timeout=10)
        assert cache.set("k2", "v2", timeout=20)
        assert cache.set("k3", "v3", timeout=30)
        assert cache._file_count == 3
        cache._prune()
        # the record with the earliest expiry is evicted from the bucket
        assert cache._file_count == 2
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"
        # the surviving records still share a single bucket file
        assert len(list(cache._list_dir())) == 1

    def test_collision_safe_write_failure_preserves_records(self, tmpdir):
        """A failed write leaves the existing bucket records intact."""
        cache = FileSystemCache(
            tmpdir, hash_method=constant_hash, collision_safe=True
        )
        assert cache.set("alpha", 1)
        assert cache.set("beta", 2)
        with patch("cachelib.file.os.replace", side_effect=OSError("boom")):
            assert cache.set("gamma", 3) is False
        assert cache.get("alpha") == 1
        assert cache.get("beta") == 2
        assert cache.get("gamma") is None
        assert cache._file_count == 2

    def test_collision_safe_legacy_file_not_adopted(self, tmpdir):
        """Legacy files lack the original key, so they are never matched,
        deleted by key, or merged into a bucket."""
        legacy = FileSystemCache(tmpdir, hash_method=constant_hash)
        assert legacy.set("ghost", "old")
        cache = FileSystemCache(
            tmpdir, hash_method=constant_hash, collision_safe=True
        )
        # ownership cannot be verified: no hit, no delete
        assert cache.get("ghost") is None
        assert cache.has("ghost") is False
        assert cache.delete("ghost") is True
        assert legacy.get("ghost") == "old"
        # a set to the same address replaces the unmigratable record
        assert cache.set("real", "new")
        assert cache.get("real") == "new"
        assert cache._file_count == 1
