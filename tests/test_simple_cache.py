import threading
from time import sleep

import pytest
from clear import ClearTests
from common import CommonTests
from has import HasTests
from serializer import SerializerTests

from cachelib import BorrowHandle
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

    def test_borrow_delays_cleanup_until_final_release(self):
        cleaned = []
        cache = self.cache_factory(value_cleanup=cleaned.append)
        old_value = object()
        new_value = "new-value"
        cache.set("key", old_value, timeout=1)

        first_borrow = cache.borrow("key")
        second_borrow = cache.borrow("key")
        assert first_borrow is not None
        assert second_borrow is not None
        assert first_borrow.value is old_value
        assert second_borrow.value is old_value

        sleep(1.1)
        assert cache.borrow("key") is None
        assert cache.get("key") is None
        assert cache.has("key") is False
        assert cache.set("key", new_value, timeout=0) is True
        assert cache.get("key") == new_value
        assert cleaned == []

        first_borrow.release()
        assert cleaned == []
        first_borrow.release()
        assert cleaned == []

        second_borrow.release()
        assert cleaned == [old_value]
        second_borrow.release()
        assert cleaned == [old_value]

        cache.delete("key")
        assert cleaned == [old_value, new_value]

    def test_borrow_handle_is_context_manager_and_supports_close(self):
        cleaned = []
        cache = self.cache_factory(value_cleanup=cleaned.append)
        value = object()
        cache.set("key", value)

        with cache.borrow("key") as handle:
            assert isinstance(handle, BorrowHandle)
            assert handle.value is value
        assert cleaned == []

        handle = cache.borrow("key")
        assert handle is not None
        cache.delete("key")
        handle.close()
        assert cleaned == [value]

    def test_rewrite_keeps_independent_borrow_counts(self):
        cleaned = []
        cache = self.cache_factory(threshold=2, value_cleanup=cleaned.append)
        old_value = object()
        new_value = object()
        cache.set("key", old_value)
        old_borrow = cache.borrow("key")
        assert old_borrow is not None

        assert cache.set("key", new_value) is True
        extra_new_borrow = cache.borrow("key")
        assert extra_new_borrow is not None
        new_borrow = cache.borrow("key")
        assert new_borrow is not None
        assert new_borrow.value is new_value

        extra_new_borrow.release()
        new_borrow.release()
        assert cleaned == []
        old_borrow.release()
        assert cleaned == [old_value]

        cache.delete("key")
        assert cleaned == [old_value, new_value]

    def test_borrowed_capacity_rejects_new_values(self):
        cleaned = []
        cache = self.cache_factory(threshold=1, value_cleanup=cleaned.append)
        value = object()
        cache.set("key", value, timeout=1)
        borrow = cache.borrow("key")
        assert borrow is not None

        assert cache.set("other", object()) is False
        assert cache.add("other", object()) is False
        assert cache.has("other") is False
        assert cleaned == []

        sleep(1.1)
        assert cache.set("other", object(), timeout=0) is False
        borrow.release()
        assert cleaned == [value]

    def test_expired_entry_seen_by_get_is_cleaned_once(self):
        cleaned = []
        cache = self.cache_factory(value_cleanup=cleaned.append)
        value = object()
        cache.set("key", value, timeout=1)

        sleep(1.1)
        assert cache.get("key") is None
        assert cleaned == [value]
        assert cache.get("key") is None
        assert cache.has("key") is False
        assert cleaned == [value]

    def test_clear_retires_borrowed_versions_and_cleans_unborrowed_once(self):
        cleaned = []
        cache = self.cache_factory(value_cleanup=cleaned.append)
        borrowed_value = object()
        unborrowed_value = object()
        cache.set("borrowed", borrowed_value)
        cache.set("unborrowed", unborrowed_value)
        borrow = cache.borrow("borrowed")
        assert borrow is not None

        assert cache.clear() is True
        assert cleaned == [unborrowed_value]
        assert cache.get("borrowed") is None

        borrow.release()
        assert cleaned == [unborrowed_value, borrowed_value]
        borrow.release()
        assert cleaned == [unborrowed_value, borrowed_value]

    def test_cleanup_errors_are_collected_without_losing_other_values(self):
        first = object()
        second = object()
        third = object()
        cleaned = []

        def cleanup(value):
            cleaned.append(value)
            if value is second:
                raise RuntimeError("second cleanup failed")

        cache = self.cache_factory(value_cleanup=cleanup)
        cache.set("first", first)
        cache.set("second", second)
        cache.set("third", third)

        with pytest.raises(ExceptionGroup) as exc_info:
            cache.clear()
        assert len(exc_info.value.exceptions) == 1
        assert str(exc_info.value.exceptions[0]) == "second cleanup failed"
        assert cleaned == [first, second, third]

        borrow = cache.borrow("first")
        assert borrow is None
        assert cache.clear() is True
        assert cleaned == [first, second, third]

    def test_failed_borrow_cleanup_is_not_retried(self):
        calls = []

        def cleanup(value):
            calls.append(value)
            raise RuntimeError("cleanup failed")

        cache = self.cache_factory(value_cleanup=cleanup)
        value = object()
        cache.set("key", value)
        handle = cache.borrow("key")
        assert handle is not None
        cache.delete("key")

        with pytest.raises(ExceptionGroup, match="cleanup failed"):
            handle.release()
        assert calls == [value]

        handle.release()
        assert calls == [value]

    def test_cleanup_keyword_aliases_value_cleanup(self):
        cleaned = []
        cache = self.cache_factory(cleanup=cleaned.append)
        value = object()
        cache.set("key", value)
        cache.delete("key")
        assert cleaned == [value]
