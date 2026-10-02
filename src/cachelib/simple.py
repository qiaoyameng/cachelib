import datetime as dt
import threading
import typing as _t
from dataclasses import dataclass
from time import time

from cachelib.base import BaseCache
from cachelib.serializers import SimpleSerializer


ValueCleanupCallable = _t.Callable[[_t.Any], None]


@dataclass(eq=False, slots=True)
class _Entry:
    key: str
    expires: int
    value: _t.Any
    serialized: bytes | None
    cleanup: ValueCleanupCallable | None = None
    borrowed_count: int = 0
    retired: bool = False
    cleaned: bool = False

    def is_valid(self, now: float) -> bool:
        return self.expires == 0 or self.expires > now


class BorrowHandle:
    """A handle keeping a cached value alive after it is retired."""

    __slots__ = ("_cache", "_entry", "_key", "_released")

    def __init__(self, cache: "SimpleCache", key: str, entry: _Entry) -> None:
        self._cache = cache
        self._key = key
        self._entry = entry
        self._released = False

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> _t.Any:
        return self._entry.value

    def release(self) -> None:
        """Release this borrow. Safe to call more than once."""
        self._cache._release_handle(self)

    def close(self) -> None:
        self.release()

    def __enter__(self) -> "BorrowHandle":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.release()
        return False


BorrowedValue = BorrowHandle


class SimpleCache(BaseCache):
    """Simple memory cache for single process environments. All operations
    are protected by a :class:`threading.RLock`, making a cache instance safe
    to use from multiple threads within the same process.

    :param threshold: the maximum number of item versions retained by the
        cache. Retired versions with active borrows continue to count toward
        this threshold.
    :param default_timeout: the default timeout that is used if no timeout is
        specified on :meth:`~.BaseCache.set`. Either a number of seconds or a
        :class:`datetime.timedelta`. A timeout of
        0 indicates that the cache never expires.

        .. versionchanged:: 0.17.0
            Accepts a :class:`datetime.timedelta`.
    :param ignore_delete_many_errors: If False, delete_many() will raise
        a RuntimeError if any key fails to delete. Keys that do not
        exist are considered successfully deleted and do not raise.

        .. versionadded:: 0.16.0
    :param value_cleanup: Called once with a cached value after it is retired
        and all borrow handles for that version have been released.
    """

    serializer = SimpleSerializer()

    def __init__(
        self,
        threshold: int = 500,
        default_timeout: int | dt.timedelta = 300,
        ignore_delete_many_errors: bool = True,
        value_cleanup: ValueCleanupCallable | None = None,
    ) -> None:
        BaseCache.__init__(
            self, default_timeout, ignore_delete_many_errors=ignore_delete_many_errors
        )
        self._cache: dict[str, _Entry] = {}
        self._retired: set[_Entry] = set()
        self._threshold = threshold or 500  # threshold = 0
        self._lock = threading.RLock()
        self._value_cleanup = value_cleanup

    def _retained_count(self) -> int:
        return len(self._cache) + len(self._retired)

    @staticmethod
    def _run_cleanups(pending: list[tuple[ValueCleanupCallable, _t.Any]]) -> None:
        errors: list[Exception] = []
        for cleanup, value in pending:
            try:
                cleanup(value)
            except Exception as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("cached value cleanup failed", errors)

    def _cleanup_entry(
        self,
        entry: _Entry,
        pending: list[tuple[ValueCleanupCallable, _t.Any]],
    ) -> None:
        if entry.cleaned:
            return
        entry.cleaned = True
        self._retired.discard(entry)
        if entry.cleanup is not None:
            pending.append((entry.cleanup, entry.value))

    def _retire_entry(
        self,
        key: str,
        entry: _Entry,
        pending: list[tuple[ValueCleanupCallable, _t.Any]],
    ) -> None:
        if entry.retired:
            return
        entry.retired = True
        if self._cache.get(key) is entry:
            self._cache.pop(key, None)
        if entry.borrowed_count:
            self._retired.add(entry)
        else:
            self._cleanup_entry(entry, pending)

    def _remove_expired(
        self, now: float, pending: list[tuple[ValueCleanupCallable, _t.Any]]
    ) -> None:
        for key, entry in list(self._cache.items()):
            if not entry.is_valid(now):
                self._retire_entry(key, entry, pending)

    def _make_room(
        self,
        key: str,
        replacing_unborrowed: bool,
        pending: list[tuple[ValueCleanupCallable, _t.Any]],
    ) -> bool:
        now = time()
        self._remove_expired(now, pending)

        while (
            self._retained_count() + (0 if replacing_unborrowed else 1)
            > self._threshold
        ):
            candidates = [
                entry
                for entry_key, entry in self._cache.items()
                if entry_key != key and entry.borrowed_count == 0
            ]
            if not candidates:
                return False
            oldest = min(candidates, key=lambda entry: entry.expires)
            self._retire_entry(oldest.key, oldest, pending)
        return True

    def _release_handle(self, handle: BorrowHandle) -> None:
        if handle._released:
            return
        pending: list[tuple[ValueCleanupCallable, _t.Any]] = []
        with self._lock:
            if handle._released:
                return
            handle._released = True
            entry = handle._entry
            entry.borrowed_count -= 1
            if entry.borrowed_count == 0 and entry.retired:
                self._cleanup_entry(entry, pending)
        self._run_cleanups(pending)

    def _normalize_timeout(self, timeout: int | dt.timedelta | None) -> int:
        normalized_timeout = BaseCache._normalize_timeout(self, timeout)
        if normalized_timeout > 0:
            normalized_timeout = int(time()) + normalized_timeout
        return normalized_timeout

    def get(self, key: str) -> _t.Any:
        pending: list[tuple[ValueCleanupCallable, _t.Any]] = []
        with self._lock:
            now = time()
            try:
                entry = self._cache[key]
            except KeyError:
                return None
            if entry.is_valid(now):
                if entry.serialized is None:
                    return None
                return self.serializer.loads(entry.serialized)
            self._retire_entry(key, entry, pending)
        self._run_cleanups(pending)
        return None

    def _set(
        self,
        key: str,
        value: _t.Any,
        timeout: int | dt.timedelta | None,
        add: bool,
    ) -> bool:
        pending: list[tuple[ValueCleanupCallable, _t.Any]] = []
        with self._lock:
            now = time()
            expires = self._normalize_timeout(timeout)
            old = self._cache.get(key)
            if old is not None and not old.is_valid(now):
                self._retire_entry(key, old, pending)
                old = None

            if add and old is not None:
                accepted = False
            else:
                replacing_unborrowed = old is not None and old.borrowed_count == 0
                accepted = self._make_room(key, replacing_unborrowed, pending)
                if accepted:
                    serialized = self.serializer.dumps(value)
                    if old is not None:
                        self._retire_entry(key, old, pending)
                    self._cache[key] = _Entry(
                        key=key,
                        expires=expires,
                        value=value,
                        serialized=serialized,
                        cleanup=self._value_cleanup,
                    )
        self._run_cleanups(pending)
        return accepted

    def set(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool | None:
        return self._set(key, value, timeout, add=False)

    def add(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool:
        return self._set(key, value, timeout, add=True)

    def borrow(self, key: str) -> BorrowHandle | None:
        """Return a handle for the current value, or ``None`` if unavailable."""
        pending: list[tuple[ValueCleanupCallable, _t.Any]] = []
        with self._lock:
            now = time()
            try:
                entry = self._cache[key]
            except KeyError:
                return None
            if not entry.is_valid(now):
                self._retire_entry(key, entry, pending)
                handle: BorrowHandle | None = None
            else:
                entry.borrowed_count += 1
                handle = BorrowHandle(self, key, entry)
        self._run_cleanups(pending)
        return handle

    acquire = borrow

    def release_borrow(self, handle: BorrowHandle) -> None:
        handle.release()

    release_handle = release_borrow

    def delete(self, key: str) -> bool:
        pending: list[tuple[ValueCleanupCallable, _t.Any]] = []
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry is None:
                return False
            self._retire_entry(key, entry, pending)
        self._run_cleanups(pending)
        return True

    def has(self, key: str) -> bool:
        pending: list[tuple[ValueCleanupCallable, _t.Any]] = []
        with self._lock:
            now = time()
            try:
                entry = self._cache[key]
            except KeyError:
                return False
            if entry.is_valid(now):
                return True
            self._retire_entry(key, entry, pending)
        self._run_cleanups(pending)
        return False

    def clear(self) -> bool:
        pending: list[tuple[ValueCleanupCallable, _t.Any]] = []
        with self._lock:
            entries = [*self._cache.values(), *self._retired]
            self._cache.clear()
            self._retired.clear()
            for entry in entries:
                if entry.cleaned:
                    continue
                entry.retired = True
                if entry.borrowed_count:
                    self._retired.add(entry)
                else:
                    self._cleanup_entry(entry, pending)
        self._run_cleanups(pending)
        return True

    def inc(self, key: str, delta: int = 1) -> int | None:
        with self._lock:
            return super().inc(key, delta)

    def dec(self, key: str, delta: int = 1) -> int | None:
        with self._lock:
            return super().dec(key, delta)
