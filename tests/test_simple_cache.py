import threading
from time import sleep

import pytest
from clear import ClearTests
from common import CommonTests
from has import HasTests
from serializer import SerializerTests

from cachelib import SimpleCache
from cachelib.serializers import BaseSerializer


class SillySerializer(BaseSerializer):
    """A pointless serializer only for testing"""

    def dumps(self, value):
        return repr(value).encode()

    def loads(self, bvalue):
        return eval(bvalue.decode())


class CustomCache(SimpleCache):
    """Our custom cache client with non-default serializer"""

    # overwrite serializer
    serializer = SillySerializer()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Resource:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True, params=[SimpleCache, CustomCache])
def cache_factory(request):
    def _factory(self, *args, **kwargs):
        return request.param(*args, **kwargs)

    request.cls.cache_factory = _factory


class TestSimpleCache(CommonTests, HasTests, ClearTests, SerializerTests):
    def test_threshold(self):
        threshold = len(self.sample_pairs) // 2
        cache = self.cache_factory(threshold=threshold)
        assert cache.set_many(self.sample_pairs)
        assert abs(len(cache._cache) - threshold) <= 1

    def test_prune_old_entries(self):
        threshold = 2 * len(self.sample_pairs) - 1
        cache = self.cache_factory(threshold=threshold)
        for k, v in self.sample_pairs.items():
            assert cache.set(f"{k}-t1", v, timeout=1)
            assert cache.set(f"{k}-t5", v, timeout=5)
        sleep(2)
        for k, v in self.sample_pairs.items():
            assert cache.set(k, v)
            assert f"{k}-t5" in cache._cache.keys()
            assert f"{k}-t1" not in cache._cache.keys()

    def test_threshold_zero_defaults_to_500(self):
        """SimpleCache(threshold=0) should silently use 500, not 0."""
        cache = self.cache_factory(threshold=0)
        assert cache._threshold == 500

    def test_add_on_expired_key_succeeds(self):
        """add() on a key whose TTL has elapsed should succeed."""
        cache = self.cache_factory()
        cache.set("key", "original", timeout=1)
        sleep(2)
        assert cache.add("key", "new") is True
        assert cache.get("key") == "new"

    def test_has_on_expired_key_returns_false(self):
        cache = self.cache_factory()
        cache.set("key", "value", timeout=1)
        sleep(2)
        assert cache.has("key") is False

    def test_delete_nonexistent_returns_false(self):
        cache = self.cache_factory()
        assert cache.delete("does_not_exist") is False

    def test_concurrent_set_no_data_corruption(self):
        """Concurrent writes of distinct keys must preserve every value."""
        cache = self.cache_factory()
        errors = []
        result_lock = threading.Lock()
        num_threads = 10
        writes_per_thread = 50
        barrier = threading.Barrier(num_threads)
        expected = {
            f"thread-{thread_id}-key-{i}": thread_id * 1000 + i
            for thread_id in range(num_threads)
            for i in range(writes_per_thread)
        }

        def writer(thread_id: int) -> None:
            barrier.wait()

            for i in range(writes_per_thread):
                key = f"thread-{thread_id}-key-{i}"
                try:
                    cache.set(key, expected[key])
                except Exception as exc:
                    with result_lock:
                        errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(t,)) for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised during concurrent set: {errors}"
        assert len(cache._cache) == len(expected)
        assert cache.get_dict(*expected.keys()) == expected

    def test_concurrent_inc_is_consistent(self):
        """Concurrent increments on a single key must not lose updates."""
        cache = self.cache_factory()

        class _SlowSerializer:
            def __init__(self, inner):
                self._inner = inner

            def dumps(self, value):
                sleep(0.0001)
                return self._inner.dumps(value)

            def loads(self, bvalue):
                return self._inner.loads(bvalue)

        cache.serializer = _SlowSerializer(type(cache).serializer)
        cache.set("counter", 0)
        num_threads = 20
        increments_each = 50
        barrier = threading.Barrier(num_threads)

        def incrementer() -> None:
            barrier.wait()
            for _ in range(increments_each):
                cache.inc("counter", 1)

        threads = [threading.Thread(target=incrementer) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache.get("counter") == num_threads * increments_each

    def test_concurrent_add_only_one_winner(self):
        """Only one thread must win the race on add() for a fresh key."""
        cache = self.cache_factory()
        results = []
        lock = threading.Lock()

        def adder() -> None:
            result = cache.add("race-key", 1)
            with lock:
                results.append(result)

        threads = [threading.Thread(target=adder) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [r for r in results if r is True]
        assert len(winners) == 1, f"Expected exactly 1 winner, got {len(winners)}"

    def test_concurrent_get_set_preserves_valid_shared_values(self):
        """Concurrent readers must only observe complete values."""
        cache = self.cache_factory()

        class _SlowSerializer:
            def __init__(self, inner):
                self._inner = inner

            def dumps(self, value):
                sleep(0.0001)
                return self._inner.dumps(value)

            def loads(self, bvalue):
                sleep(0.0001)
                return self._inner.loads(bvalue)

        cache.serializer = _SlowSerializer(type(cache).serializer)

        initial_value = ("initial", -1)
        cache.set("shared", initial_value)
        errors = []
        observed_values = []
        result_lock = threading.Lock()
        num_readers = 5
        num_writers = 5
        writes_per_writer = 100
        barrier = threading.Barrier(num_readers + num_writers)
        written_values = {
            (thread_id, i)
            for thread_id in range(num_writers)
            for i in range(writes_per_writer)
        }
        expected_values = {initial_value} | written_values

        def reader() -> None:
            barrier.wait()

            for _ in range(writes_per_writer):
                try:
                    value = cache.get("shared")
                    with result_lock:
                        observed_values.append(value)
                except Exception as exc:
                    with result_lock:
                        errors.append(exc)

        def writer(thread_id: int) -> None:
            barrier.wait()

            for i in range(writes_per_writer):
                try:
                    cache.set("shared", (thread_id, i))
                except Exception as exc:
                    with result_lock:
                        errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(num_readers)]
        threads += [
            threading.Thread(target=writer, args=(thread_id,))
            for thread_id in range(num_writers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions during concurrent get/set: {errors}"
        assert observed_values
        assert set(observed_values) <= expected_values
        assert len(cache._cache) == 1
        assert cache.has("shared") is True
        assert cache.get("shared") in written_values

    def test_borrow_delays_cleanup_until_expired_version_is_released(self):
        cleaned = []
        cache = self.cache_factory(
            threshold=2, value_cleanup=lambda value: cleaned.append(value)
        )
        old = Resource("old")
        new = Resource("new")

        handle = cache.borrow("key")
        assert handle is None

        cache.set("key", old, timeout=1)
        handle = cache.borrow("key")
        assert handle is not None
        assert handle.value is old
        sleep(2)

        assert cache.has("key") is False
        assert cache.get("key") is None
        assert cache.set("key", new) is True
        other_new_handle = cache.borrow("key")
        assert other_new_handle is not None
        new_handle = cache.borrow("key")
        assert new_handle is not None
        assert new_handle.value is new
        assert cleaned == []

        handle.release()
        handle.release()
        assert cleaned == [old]

        new_handle.release()
        assert cleaned == [old]
        assert other_new_handle.value is new

        cache.delete("key")
        other_new_handle.release()
        assert cleaned == [old, new]

    def test_borrowed_versions_have_independent_counts(self):
        cleaned = []
        cache = self.cache_factory(
            threshold=3, value_cleanup=lambda value: cleaned.append(value)
        )
        first = Resource("first")
        second = Resource("second")

        cache.set("key", first)
        first_handle = cache.borrow("key")
        assert first_handle is not None
        assert cache.set("key", second) is True
        second_handle = cache.borrow("key")
        assert second_handle is not None

        first_handle.release()
        assert cleaned == [first]
        cache.delete("key")
        second_handle.release()
        assert cleaned == [first, second]

    def test_borrowed_version_rejects_new_write_when_capacity_is_full(self):
        cleaned = []
        cache = self.cache_factory(
            threshold=1, value_cleanup=lambda value: cleaned.append(value)
        )
        old = Resource("old")
        new = Resource("new")

        cache.set("key", old)
        handle = cache.borrow("key")
        assert handle is not None
        assert cache.set("key", new) is False
        assert cache.has("key") is True

        other = Resource("other")
        assert cache.set("other", other) is False
        assert "other" not in cache._cache

        handle.release()
        assert cleaned == []
        assert cache.set("key", new) is True
        assert cleaned == [old]
        new_handle = cache.borrow("key")
        assert new_handle is not None
        assert new_handle.value is new

    def test_cleanup_error_does_not_clean_twice_or_skip_other_version(self):
        def cleanup(value):
            cleaned.append(value)
            if value is old:
                raise RuntimeError("boom")

        cleaned = []
        cache = self.cache_factory(threshold=2, value_cleanup=cleanup)
        old = Resource("old")
        new = Resource("new")

        old_handle = cache.borrow("missing")
        assert old_handle is None
        cache.set("key", old)
        old_handle = cache.borrow("key")
        assert old_handle is not None
        cache.delete("key")
        cache.set("key", new)
        new_handle = cache.borrow("key")
        assert new_handle is not None

        with pytest.raises(RuntimeError, match="boom"):
            old_handle.release()
        old_handle.release()
        assert cleaned == [old]

        cache.clear()
        assert cleaned == [old]
        new_handle.release()
        assert cleaned == [old, new]
        cache.clear()
        assert cleaned == [old, new]

    def test_clear_cleans_each_borrowed_version_once(self):
        cleaned = []
        cache = self.cache_factory(
            threshold=2, value_cleanup=lambda value: cleaned.append(value)
        )
        old = Resource("old")
        new = Resource("new")

        cache.set("key", old)
        old_handle = cache.borrow("key")
        assert old_handle is not None
        cache.set("key", new)
        new_handle = cache.borrow("key")
        assert new_handle is not None

        assert cache.clear() is True
        assert cache.get("key") is None
        assert cleaned == []

        old_handle.release()
        new_handle.release()
        old_handle.release()
        new_handle.release()
        assert cleaned == [old, new]

    def test_borrow_handle_supports_context_manager(self):
        cleaned = []
        cache = self.cache_factory(
            value_cleanup=lambda value: cleaned.append(value)
        )
        value = Resource("context")

        cache.set("key", value)
        with cache.borrow("key") as handle:
            assert handle.value is value
            cache.delete("key")
            assert cleaned == []
        assert cleaned == [value]

    def test_borrow_preserves_original_external_resource(self):
        resource = Resource("external")
        cache = self.cache_factory(threshold=2, value_cleanup=Resource.close)

        cache.set("resource", resource)
        handle = cache.borrow("resource")
        assert handle is not None
        assert handle.value is resource
        cache.delete("resource")
        assert resource.closed is False

        replacement = Resource("replacement")
        assert cache.set("resource", replacement) is True
        cache.delete("resource")
        handle.release()
        handle.release()
        assert resource.closed is True
        assert replacement.closed is True
