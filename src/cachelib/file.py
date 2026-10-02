import datetime as dt
import errno
import hashlib
import logging
import os
import platform
import stat
import struct
import tempfile
import typing as _t
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from time import time

from cachelib.base import BaseCache
from cachelib.serializers import FileSystemSerializer


class FileSystemCache(BaseCache):
    """A cache that stores the items on the file system.  This cache depends
    on being the only user of the ``cache_dir``.  Make absolutely sure that
    nobody but this cache stores files there or otherwise the cache will
    randomly delete files therein.

    :param cache_dir: the directory where cache files are stored.
    :param threshold: the maximum number of items the cache stores before
        it starts deleting some. A threshold value of 0
        indicates no threshold.
    :param default_timeout: the default timeout that is used if no timeout is
        specified on :meth:`~.BaseCache.set`. Either a number of seconds or a
        :class:`datetime.timedelta`. A timeout of
        0 indicates that the cache never expires.

        .. versionchanged:: 0.17.0
            Accepts a :class:`datetime.timedelta`.
    :param mode: the file mode wanted for the cache files, default 0600
    :param hash_method: Default :func:`~hashlib.sha256`. The hash method used to
        generate the filename for cached results.
    :param ignore_delete_many_errors: If False, delete_many() will raise
        a RuntimeError if any key fails to delete. Keys that do not
        exist are considered successfully deleted and do not raise.

        .. versionadded:: 0.16.0
    :param collision_safe: If True, use a collision-safe storage format.
        Every record stores the full original key, and all records that
        hash to the same filename are kept together in a single "bucket"
        file, so keys that collide under ``hash_method`` never overwrite,
        false-hit, or delete each other. Entry counting and expiry
        pruning are based on the records stored inside each bucket, not
        on file names.

        Legacy cache files (written with ``collision_safe=False``) do not
        contain the original key, so they cannot be migrated into the
        bucket format: their ownership cannot be verified and is never
        guessed. They are treated as foreign files - ``get`` and ``has``
        miss them, ``delete`` leaves them untouched, and a ``set`` to the
        same hash address replaces them. Switching an existing cache
        directory to collision-safe mode therefore effectively
        invalidates old entries; use a fresh directory or
        :meth:`~.BaseCache.clear` first.

        .. versionadded:: 0.18.0
    """

    #: used for temporary files by the FileSystemCache
    _fs_transaction_suffix = ".__wz_cache"
    #: keep amount of files in a cache element
    _fs_count_file = "__wz_cache_count"
    #: magic marker identifying the collision-safe bucket format
    _fs_bucket_magic = b"CLB1"

    serializer = FileSystemSerializer()

    def __init__(
        self,
        cache_dir: str,
        threshold: int = 500,
        default_timeout: int | dt.timedelta = 300,
        mode: int | None = None,
        hash_method: _t.Any = hashlib.sha256,
        ignore_delete_many_errors: bool = True,
        collision_safe: bool = False,
    ):
        BaseCache.__init__(
            self, default_timeout, ignore_delete_many_errors=ignore_delete_many_errors
        )
        self._path = cache_dir
        self._threshold = threshold
        self._hash_method = hash_method
        self._collision_safe = collision_safe

        # Mode set by user takes precedence. If no mode has
        # been given, we need to set the correct default based
        # on user platform.
        self._mode = mode
        if self._mode is None:
            self._mode = self._get_compatible_platform_mode()

        try:
            os.makedirs(self._path)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise

        # If there are many files and a zero threshold,
        # the list_dir can slow initialisation massively
        if self._threshold != 0:
            if self._collision_safe:
                # Count entries, not files: a bucket file may hold many
                # records and the entry count cannot be inferred from
                # file names alone.
                self._update_count(value=self._count_entries())
            else:
                self._update_count(value=len(list(self._list_dir())))

    def _get_compatible_platform_mode(self) -> int:
        mode = 0o600  # nix systems
        if platform.system() == "Windows":
            mode = stat.S_IWRITE
        return mode

    @property
    def _file_count(self) -> int:
        return self.get(self._fs_count_file) or 0

    def _update_count(self, delta: int | None = None, value: int | None = None) -> None:
        # If we have no threshold, don't count files
        if self._threshold == 0:
            return
        if delta is not None:
            new_count = self._file_count + delta
        else:
            new_count = value or 0
        self.set(self._fs_count_file, new_count, mgmt_element=True)

    def _normalize_timeout(self, timeout: int | dt.timedelta | None) -> int:
        normalized_timeout = BaseCache._normalize_timeout(self, timeout)
        if normalized_timeout != 0:
            normalized_timeout = int(time()) + normalized_timeout
        return int(normalized_timeout)

    def _is_mgmt(self, name: str) -> bool:
        fshash = self._get_filename(self._fs_count_file).split(os.sep)[-1]
        return name == fshash or name.endswith(self._fs_transaction_suffix)

    def _list_dir(self) -> _t.Generator[str, None, None]:
        """return a list of (fully qualified) cache filenames"""
        return (
            os.path.join(self._path, fn)
            for fn in os.listdir(self._path)
            if not self._is_mgmt(fn)
        )

    def _over_threshold(self) -> bool:
        return self._threshold != 0 and self._file_count > self._threshold

    def _count_entries(self) -> int:
        """Count stored entries by inspecting file contents. A bucket file
        contributes as many entries as it holds records; any other file
        (legacy format or unreadable) counts as a single entry."""
        total = 0
        for fname in self._list_dir():
            try:
                records = self._load_bucket(fname)
            except (OSError, EOFError, struct.error):
                total += 1
            else:
                total += 1 if records is None else len(records)
        return total

    def _load_bucket(
        self, filename: str
    ) -> list[tuple[int, bytes, _t.Any]] | None:
        """Read a collision-safe bucket file.

        Returns a list of ``(expires, key_bytes, value)`` records, or
        ``None`` if the file is not in bucket format (e.g. a legacy cache
        file, whose original key is not stored and cannot be verified).
        Raises :exc:`FileNotFoundError` if the file does not exist and
        :exc:`EOFError`/:exc:`struct.error` if the content is truncated.
        """
        with self._safe_stream_open(filename, "rb") as f:
            if f.read(len(self._fs_bucket_magic)) != self._fs_bucket_magic:
                return None
            records = []
            while True:
                header = f.read(8)
                if not header:
                    break
                expires, key_len = struct.unpack("II", header)
                bkey = f.read(key_len)
                if len(bkey) != key_len:
                    raise EOFError("truncated key in bucket file")
                records.append((expires, bkey, self.serializer.load(f)))
        return records

    def _dump_bucket(
        self, filename: str, records: list[tuple[int, bytes, _t.Any]]
    ) -> int:
        """Atomically write records to a bucket file and return its size.
        The previous bucket file is left untouched if writing fails."""
        fd, tmp = tempfile.mkstemp(suffix=self._fs_transaction_suffix, dir=self._path)
        with os.fdopen(fd, "wb") as f:
            f.write(self._fs_bucket_magic)
            for expires, bkey, value in records:
                f.write(struct.pack("I", expires))
                f.write(struct.pack("I", len(bkey)))
                f.write(bkey)
                self.serializer.dump(value, f)

        self._run_safely(os.replace, tmp, filename)
        self._run_safely(os.chmod, filename, self._mode)
        return Path(filename).stat().st_size

    def _remove_expired(self, now: float) -> None:
        for fname in self._list_dir():
            try:
                removed = self._remove_expired_from_file(fname, now)
            except FileNotFoundError:
                pass
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
            else:
                if removed:
                    self._update_count(delta=-removed)

    def _remove_expired_from_file(self, fname: str, now: float) -> int:
        """Remove expired records from a single cache file, returning the
        number of entries removed. Bucket files are rewritten without the
        expired records; legacy files are removed as a whole."""
        records = self._load_bucket(fname)
        if records is None:
            # Legacy format: a single record without a stored key.
            with self._safe_stream_open(fname, "rb") as f:
                expires = struct.unpack("I", f.read(4))[0]
            if expires != 0 and expires < now:
                os.remove(fname)
                return 1
            return 0
        kept = [r for r in records if r[0] == 0 or r[0] >= now]
        removed = len(records) - len(kept)
        if not removed:
            return 0
        if kept:
            self._dump_bucket(fname, kept)
        else:
            os.remove(fname)
        return removed

    def _remove_older(self) -> bool:
        entries = []
        for fname in self._list_dir():
            try:
                records = self._load_bucket(fname)
                if records is None:
                    # Legacy format: a single record without a stored key.
                    with self._safe_stream_open(fname, "rb") as f:
                        timestamp = struct.unpack("I", f.read(4))[0]
                    entries.append((timestamp, fname, None))
                else:
                    entries.extend(
                        (expires, fname, bkey) for expires, bkey, _ in records
                    )
            except FileNotFoundError:
                pass
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
        entries.sort(key=lambda item: item[0])
        removals: dict[str, list[bytes | None]] = {}
        remaining = self._file_count
        for _, fname, bkey in entries:
            if not self._threshold or remaining <= self._threshold:
                break
            removals.setdefault(fname, []).append(bkey)
            remaining -= 1
        for fname, bkeys in removals.items():
            try:
                removed = self._remove_records_from_file(fname, bkeys)
            except FileNotFoundError:
                pass
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
                return False
            else:
                self._update_count(delta=-removed)
        return True

    def _remove_records_from_file(
        self, fname: str, bkeys: list[bytes | None]
    ) -> int:
        """Remove the given records from a cache file, returning the number
        of entries removed. A ``None`` key denotes a legacy file, which is
        removed as a whole."""
        if None in bkeys:
            os.remove(fname)
            return 1
        records = self._load_bucket(fname)
        if records is None:
            os.remove(fname)
            return 1
        remove = set(bkeys)
        kept = [r for r in records if r[1] not in remove]
        removed = len(records) - len(kept)
        if kept:
            self._dump_bucket(fname, kept)
        else:
            os.remove(fname)
        return removed

    def _prune(self) -> None:
        if self._over_threshold():
            now = time()
            self._remove_expired(now)
        # if still over threshold
        if self._over_threshold():
            self._remove_older()

    def clear(self) -> bool:
        for i, fname in enumerate(self._list_dir()):
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass
            except OSError:
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
                self._update_count(delta=-i)
                return False
        self._update_count(value=0)
        return True

    def _get_filename(self, key: str | _t.Any) -> str:
        if isinstance(key, str):
            bkey = key.encode("utf-8")  # XXX unicode review
            bkey_hash = self._hash_method(bkey).hexdigest()
        else:
            raise TypeError(f"Key must be a string, received type {type(key)}")
        return os.path.join(self._path, bkey_hash)

    def get(self, key: str) -> _t.Any:
        filename = self._get_filename(key)
        if self._collision_safe:
            try:
                value, found = self._bucket_lookup(filename, key)
            except FileNotFoundError:
                pass
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    filename,
                    exc_info=True,
                )
            else:
                if found:
                    return value
            return None
        try:
            with self._safe_stream_open(filename, "rb") as f:
                pickle_time = struct.unpack("I", f.read(4))[0]
                if pickle_time == 0 or pickle_time >= time():
                    return self.serializer.load(f)
        except FileNotFoundError:
            pass
        except (OSError, EOFError, struct.error):
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )
        return None

    def _bucket_lookup(self, filename: str, key: str) -> tuple[_t.Any, bool]:
        """Return ``(value, True)`` if the bucket at ``filename`` holds a
        non-expired record whose stored key matches ``key`` exactly.
        Legacy files are never matched: without the original key their
        ownership cannot be verified."""
        records = self._load_bucket(filename)
        if records is None:
            return None, False
        bkey = key.encode("utf-8")
        now = time()
        for expires, stored_key, value in records:
            if stored_key == bkey:
                if expires == 0 or expires >= now:
                    return value, True
                return None, False
        return None, False

    def add(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool:
        filename = self._get_filename(key)
        if self._collision_safe:
            try:
                records = self._load_bucket(filename)
            except FileNotFoundError:
                records = []
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    filename,
                    exc_info=True,
                )
                return False
            if records is None:
                # A legacy file occupies this address; its key cannot be
                # verified, so conservatively refuse to overwrite it here.
                return False
            bkey = key.encode("utf-8")
            if any(stored_key == bkey for _, stored_key, _ in records):
                return False
            return self.set(key, value, timeout)
        if not os.path.exists(filename):
            return self.set(key, value, timeout)
        return False

    def set(
        self,
        key: str,
        value: _t.Any,
        timeout: int | dt.timedelta | None = None,
        mgmt_element: bool = False,
    ) -> bool:
        # Management elements have no timeout
        if mgmt_element:
            timeout = 0
        # Don't prune on management element update, to avoid loop
        else:
            self._prune()

        normalized_timeout = self._normalize_timeout(timeout)
        filename = self._get_filename(key)
        if self._collision_safe:
            return self._set_collision_safe(
                filename, key, value, normalized_timeout, mgmt_element
            )
        overwrite = os.path.isfile(filename)

        try:
            fd, tmp = tempfile.mkstemp(
                suffix=self._fs_transaction_suffix, dir=self._path
            )
            with os.fdopen(fd, "wb") as f:
                f.write(struct.pack("I", normalized_timeout))
                self.serializer.dump(value, f)

            self._run_safely(os.replace, tmp, filename)
            self._run_safely(os.chmod, filename, self._mode)

            fsize = Path(filename).stat().st_size
        except OSError:
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )
            return False
        else:
            # Management elements should not count towards threshold
            if not overwrite and not mgmt_element:
                self._update_count(delta=1)
            return fsize > 0  # function should fail if file is empty

    def _set_collision_safe(
        self,
        filename: str,
        key: str,
        value: _t.Any,
        normalized_timeout: int,
        mgmt_element: bool,
    ) -> bool:
        bkey = key.encode("utf-8")
        try:
            records = self._load_bucket(filename)
            # A legacy file counts as a single entry.
            previous = len(records) if records is not None else 1
        except FileNotFoundError:
            records, previous = [], 0
        except (OSError, EOFError, struct.error):
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )
            return False
        if records is None:
            # Legacy file: the original key is not stored, so the record
            # cannot be adopted into the bucket and is replaced.
            records = []
        records = [r for r in records if r[1] != bkey]
        records.append((normalized_timeout, bkey, value))
        try:
            fsize = self._dump_bucket(filename, records)
        except OSError:
            # The previous bucket file is left untouched on failure.
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )
            return False
        # Management elements should not count towards threshold
        if not mgmt_element:
            self._update_count(delta=len(records) - previous)
        return fsize > 0  # function should fail if file is empty

    def delete(self, key: str, mgmt_element: bool = False) -> bool:
        if self._collision_safe:
            return self._delete_collision_safe(key, mgmt_element)
        try:
            os.remove(self._get_filename(key))
        except FileNotFoundError:  # if file doesn't exist we consider it deleted
            return True
        except OSError:
            logging.warning("Exception raised while handling cache file", exc_info=True)
            return False
        else:
            # Management elements should not count towards threshold
            if not mgmt_element:
                self._update_count(delta=-1)
            return True

    def _delete_collision_safe(self, key: str, mgmt_element: bool = False) -> bool:
        filename = self._get_filename(key)
        try:
            records = self._load_bucket(filename)
        except FileNotFoundError:  # if file doesn't exist we consider it deleted
            return True
        except (OSError, EOFError, struct.error):
            logging.warning("Exception raised while handling cache file", exc_info=True)
            return False
        if records is None:
            # Legacy file: its key cannot be verified, so it must not be
            # deleted on behalf of a key it may not belong to.
            return True
        bkey = key.encode("utf-8")
        kept = [r for r in records if r[1] != bkey]
        if len(kept) == len(records):
            return True  # key not present, consider it deleted
        try:
            if kept:
                self._dump_bucket(filename, kept)
            else:
                os.remove(filename)
        except OSError:
            logging.warning("Exception raised while handling cache file", exc_info=True)
            return False
        # Management elements should not count towards threshold
        if not mgmt_element:
            self._update_count(delta=-1)
        return True

    def has(self, key: str) -> bool:
        filename = self._get_filename(key)
        if self._collision_safe:
            try:
                _, found = self._bucket_lookup(filename, key)
            except FileNotFoundError:  # if there is no file there is no key
                return False
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    filename,
                    exc_info=True,
                )
                return False
            return found
        try:
            with self._safe_stream_open(filename, "rb") as f:
                pickle_time = struct.unpack("I", f.read(4))[0]
                if pickle_time == 0 or pickle_time >= time():
                    return True
                else:
                    return False
        except FileNotFoundError:  # if there is no file there is no key
            return False
        except (OSError, EOFError, struct.error):
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )
            return False

    def _run_safely(
        self, fn: _t.Callable[..., _t.Any], *args: _t.Any, **kwargs: _t.Any
    ) -> _t.Any:
        """On some filesystems (e.g. SMB/CIFS on Linux, NTFS on Windows)
        os.replace, os.chmod and open can yield PermissionError when executed
        by two different processes concurrently. Retries with exponential
        backoff on all platforms."""
        output = None
        wait_step = 0.001
        max_sleep_time = 10.0
        total_sleep_time = 0.0

        while total_sleep_time < max_sleep_time:
            try:
                output = fn(*args, **kwargs)
            except PermissionError:
                sleep(wait_step)
                total_sleep_time += wait_step
                wait_step *= 2
            else:
                break

        return output

    @contextmanager
    def _safe_stream_open(
        self, path: str, mode: str
    ) -> _t.Generator[_t.Any, None, None]:
        fs = self._run_safely(open, path, mode)
        if fs is None:
            raise OSError
        try:
            yield fs
        finally:
            fs.close()
