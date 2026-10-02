import datetime as dt
import threading
import typing as _t
from time import time

from cachelib.base import BaseCache
from cachelib.serializers import SimpleSerializer


_ValueCleanup = _t.Callable[[_t.Any], None]


class BorrowHandle:
    """A handle to a borrowed cache value."""

    __slots__ = ("_cache", "_entry", "_released", "value")

    def __init__(
        self,
        cache: "SimpleCache",
        entry: "_CacheEntry",
        value: _t.Any,
    ) -> None:
        self._cache = cache
        self._entry = entry
        self._released = False
        self.value = value

    def release(self) -> None:
        """Release this borrow.

        Calling this more than once has no effect.
        """
        self._cache._release_borrow(self)

    def close(self) -> None:
        """Release this borrow."""
        self.release()

    def __enter__(self) -> "BorrowHandle":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class _CacheEntry:
    __slots__ = ("expires", "encoded_value", "value", "borrow_count")

    def __init__(
        self,
        expires: int,
        encoded_value: bytes | None,
        value: _t.Any,
    ) -> None:
        self.expires = expires
        self.encoded_value = encoded_value
        self.value = value
        self.borrow_count = 0


BorrowedValue = BorrowHandle


class SimpleCache(BaseCache):
    """Simple memory cache for single process environments. All operations
    are protected by a :class:`threading.RLock`, making a cache instance safe
    to use from multiple threads within the same process.

    :param threshold: the maximum number of visible and retired-but-borrowed
        items counted by the cache.
    :param default_timeout: the default timeout that is used if no timeout is
        specified on :meth:`~.BaseCache.set`. Either a number of seconds or a
        :class:`datetime.timedelta`. A timeout of
        0 indicates that the cache never expires.

        .. versionchanged:: 0.17.0
            Accepts a :class:`datetime.timedelta`.
    :param ignore_delete_many_errors: If False, delete_many() will raise
        a RuntimeError if any key fails to delete. Keys that do not
        exist are considered successfully deleted and do not raise.
    :param value_cleanup: Optional callable invoked with a cached value when
        that value version is permanently removed. Cleanup of a borrowed
        version is delayed until all of its borrow handles are released.
    """

    serializer = SimpleSerializer()

    def __init__(
        self,
        threshold: int = 500,
        default_timeout: int | dt.timedelta = 300,
        ignore_delete_many_errors: bool = True,
        value_cleanup: _ValueCleanup | None = None,
        **kwargs: _t.Any,
    ):
        cleanup = kwargs.pop("cleanup", None)
        if kwargs:
            name = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument {name!r}")
        if value_cleanup is not None and cleanup is not None:
            raise TypeError("value cleanup callback supplied more than once")
        if cleanup is not None:
            value_cleanup = cleanup
        BaseCache.__init__(
            self, default_timeout, ignore_delete_many_errors=ignore_delete_many_errors
        )
        self._cache: dict[str, _CacheEntry] = {}
        self._retired: list[_CacheEntry] = []
        self._threshold = threshold or 500
        self._lock = threading.RLock()
        self._value_cleanup = value_cleanup

    @staticmethod
    def _is_expired(entry: _CacheEntry, now: float) -> bool:
        return entry.expires != 0 and entry.expires <= now

    def _item_count(self) -> int:
        return len(self._cache) + len(self._retired)

    def _retire(
        self,
        key: str,
        entry: _CacheEntry,
        cleanups: list[_CacheEntry],
    ) -> None:
        if self._cache.get(key) is not entry:
            return
        self._cache.pop(key)
        if entry.borrow_count:
            self._retired.append(entry)
        else:
            cleanups.append(entry)

    def _remove_expired(self, now: float, cleanups: list[_CacheEntry]) -> None:
        for key, entry in list(self._cache.items()):
            if self._is_expired(entry, now):
                self._retire(key, entry, cleanups)

    def _evict_for_write(
        self,
        cleanups: list[_CacheEntry],
        additional: int = 0,
        exclude_key: str | None = None,
    ) -> bool:
        while self._item_count() + additional > self._threshold:
            candidates = [
                (entry.expires, key, entry)
                for key, entry in self._cache.items()
                if entry.borrow_count == 0 and key != exclude_key
            ]
            if not candidates:
                return False
            _, key, entry = min(candidates)
            self._cache.pop(key)
            cleanups.append(entry)
        return True

    def _run_cleanups(self, cleanups: list[_CacheEntry]) -> None:
        if self._value_cleanup is None or not cleanups:
            return
        errors: list[Exception] = []
        for entry in cleanups:
            try:
                self._value_cleanup(entry.value)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Cache value cleanup failed", errors)

    def _normalize_timeout(self, timeout: int | dt.timedelta | None) -> int:
        normalized_timeout = BaseCache._normalize_timeout(self, timeout)
        if normalized_timeout > 0:
            normalized_timeout = int(time()) + normalized_timeout
        return normalized_timeout

    def get(self, key: str) -> _t.Any:
        cleanups: list[_CacheEntry] = []
        result: _t.Any = None
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                pass
            elif self._is_expired(entry, time()):
                self._retire(key, entry, cleanups)
            elif entry.encoded_value is not None:
                result = self.serializer.loads(entry.encoded_value)
        self._run_cleanups(cleanups)
        return result

    def borrow(self, key: str) -> BorrowHandle | None:
        """Borrow the current value for ``key``.

        Return ``None`` when the key is missing or expired. The returned
        handle must be released after use.
        """
        cleanups: list[_CacheEntry] = []
        handle: BorrowHandle | None = None
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and not self._is_expired(entry, time()):
                entry.borrow_count += 1
                handle = BorrowHandle(self, entry, entry.value)
            elif entry is not None:
                self._retire(key, entry, cleanups)
        self._run_cleanups(cleanups)
        return handle

    def _release_borrow(self, handle: BorrowHandle) -> None:
        cleanups: list[_CacheEntry] = []
        with self._lock:
            if handle._released:
                return
            handle._released = True
            entry = handle._entry
            entry.borrow_count -= 1
            if entry.borrow_count == 0 and entry in self._retired:
                self._retired.remove(entry)
                cleanups.append(entry)
        self._run_cleanups(cleanups)

    def set(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool | None:
        cleanups: list[_CacheEntry] = []
        with self._lock:
            self._remove_expired(time(), cleanups)
            old_entry = self._cache.get(key)
            if old_entry is not None and old_entry.borrow_count:
                can_store = self._evict_for_write(
                    cleanups, additional=1, exclude_key=key
                )
            elif old_entry is not None:
                can_store = self._evict_for_write(cleanups, exclude_key=key)
            else:
                can_store = self._evict_for_write(cleanups, additional=1)
            if not can_store:
                success = False
            else:
                if old_entry is not None and self._cache.get(key) is old_entry:
                    self._retire(key, old_entry, cleanups)
                encoded_value = self.serializer.dumps(value)
                self._cache[key] = _CacheEntry(
                    self._normalize_timeout(timeout), encoded_value, value
                )
                success = True
        self._run_cleanups(cleanups)
        return success

    def add(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool:
        cleanups: list[_CacheEntry] = []
        with self._lock:
            self._remove_expired(time(), cleanups)
            entry = self._cache.get(key)
            if entry is not None and not self._is_expired(entry, time()):
                success = False
            elif self._evict_for_write(cleanups, additional=1):
                encoded_value = self.serializer.dumps(value)
                self._cache[key] = _CacheEntry(
                    self._normalize_timeout(timeout), encoded_value, value
                )
                success = True
            else:
                success = False
        self._run_cleanups(cleanups)
        return success

    def delete(self, key: str) -> bool:
        cleanups: list[_CacheEntry] = []
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry is None:
                return False
            if entry.borrow_count:
                self._retired.append(entry)
            else:
                cleanups.append(entry)
        self._run_cleanups(cleanups)
        return True

    def has(self, key: str) -> bool:
        cleanups: list[_CacheEntry] = []
        exists = False
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and self._is_expired(entry, time()):
                self._retire(key, entry, cleanups)
            elif entry is not None:
                exists = True
        self._run_cleanups(cleanups)
        return exists

    def clear(self) -> bool:
        cleanups: list[_CacheEntry] = []
        with self._lock:
            entries = list(self._cache.values())
            self._cache.clear()
            for entry in entries:
                if entry.borrow_count:
                    self._retired.append(entry)
                else:
                    cleanups.append(entry)
        self._run_cleanups(cleanups)
        return True

    def inc(self, key: str, delta: int = 1) -> int | None:
        with self._lock:
            return super().inc(key, delta)

    def dec(self, key: str, delta: int = 1) -> int | None:
        with self._lock:
            return super().dec(key, delta)
