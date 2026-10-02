import hashlib
import os
from time import sleep
from unittest import mock

from cachelib import FileSystemCache


def constant_hash(data=b""):
    """A hash method that maps every key to the same digest, forcing
    collisions between all keys."""
    return hashlib.sha256(b"constant-collision")


class TestCollisionSafeFileSystemCache:
    def make_cache(self, tmp_path, **kwargs):
        return FileSystemCache(
            str(tmp_path), hash_method=constant_hash, collision_safe=True, **kwargs
        )

    def test_colliding_keys_interleaved(self, tmp_path):
        cache = self.make_cache(tmp_path)
        assert cache.set("alpha", 1) is True
        assert cache.set("beta", 2) is True
        assert cache.set("gamma", 3) is True

        # all three keys share one hash but live in separate slot files
        digest = constant_hash().hexdigest()
        names = os.listdir(cache._path)
        assert digest in names
        assert f"{digest}.1" in names
        assert f"{digest}.2" in names

        assert cache.get("alpha") == 1
        assert cache.get("beta") == 2
        assert cache.get("gamma") == 3
        assert cache._file_count == 3

        # overwriting one key must not touch its neighbours
        assert cache.set("beta", 22) is True
        assert cache.get("beta") == 22
        assert cache.get("alpha") == 1
        assert cache.get("gamma") == 3
        assert cache._file_count == 3

        # add respects existing records
        assert cache.add("alpha", 100) is False
        assert cache.get("alpha") == 1
        assert cache.add("delta", 4) is True
        assert cache.get("delta") == 4
        assert cache._file_count == 4

        # has follows the same rules
        assert cache.has("gamma") is True
        assert cache.has("missing") is False

        # deleting one key must not delete or corrupt the others
        assert cache.delete("beta") is True
        assert cache.get("beta") is None
        assert cache.has("beta") is False
        assert cache.get("alpha") == 1
        assert cache.get("gamma") == 3
        assert cache.get("delta") == 4
        assert cache._file_count == 3

        # deleting an unknown key is a successful no-op
        assert cache.delete("missing") is True

        # a freed slot can be reused by a new record
        assert cache.set("beta", 2) is True
        assert cache.get("beta") == 2
        assert cache._file_count == 4

        assert cache.clear() is True
        assert cache.get("alpha") is None
        assert cache._file_count == 0

    def test_expired_colliding_entries(self, tmp_path):
        cache = self.make_cache(tmp_path, threshold=10)
        assert cache.set("short", "x", timeout=1)
        assert cache.set("long", "y", timeout=100)
        sleep(2)
        assert cache.get("short") is None
        assert cache.has("short") is False
        assert cache.get("long") == "y"

    def test_prune_with_collisions(self, tmp_path):
        cache = self.make_cache(tmp_path, threshold=2, default_timeout=100)
        for key in ("k1", "k2", "k3", "k4"):
            assert cache.set(key, key)
        # pruning ran before the last write: threshold + 1 entries remain
        assert cache._file_count == 3
        values = {key: cache.get(key) for key in ("k1", "k2", "k3", "k4")}
        # every surviving entry still maps to its own value, never a
        # neighbour's
        for key, value in values.items():
            assert value in (None, key)
        # exactly one entry was evicted to satisfy the threshold
        assert list(values.values()).count(None) == 1

    def test_legacy_files_are_not_migrated_or_guessed(self, tmp_path):
        legacy = FileSystemCache(str(tmp_path), hash_method=constant_hash)
        assert legacy.set("oldkey", "oldvalue")

        cache = self.make_cache(tmp_path)
        # the legacy record carries no key, so ownership must not be guessed
        assert cache.get("oldkey") is None
        assert cache.has("oldkey") is False
        assert cache._file_count == 0

        # writing a colliding key must not overwrite the foreign file
        assert cache.set("newkey", "newvalue")
        assert cache.get("newkey") == "newvalue"
        assert legacy.get("oldkey") == "oldvalue"

        # delete must not remove the foreign file
        assert cache.delete("oldkey") is True
        assert legacy.get("oldkey") == "oldvalue"

        # clear() removes everything, including legacy files
        assert cache.clear() is True
        assert legacy.get("oldkey") is None

    def test_failed_write_preserves_existing_record(self, tmp_path):
        cache = self.make_cache(tmp_path)
        assert cache.set("alpha", 1)

        with mock.patch("os.replace", side_effect=OSError("disk full")):
            # overwriting an existing record fails without destroying it
            assert cache.set("alpha", 2) is False
            # a new key fails without leaving a partial record behind
            assert cache.set("beta", 2) is False

        assert cache.get("alpha") == 1
        assert cache.get("beta") is None
        assert cache._file_count == 1
