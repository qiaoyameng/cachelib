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
from time import sleep
from time import time

from cachelib.base import BaseCache
from cachelib.serializers import FileSystemSerializer


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


class FileSystemCache(BaseCache):
    """A cache that stores the items on the file system.  This cache depends
    on being the only user of the ``cache_dir``.  Make absolutely sure that
    nobody but this cache stores files there or otherwise the cache will
    randomly delete files therein.

    :param cache_dir: the directory where cache files are stored.
    :param threshold: the maximum number of items the cache stores before
        it starts deleting some. A threshold value of 0
        indicates no threshold.
    :param max_bytes: the maximum total size in bytes that cache entries may
        occupy on disk, measured as the actual size of the committed cache
        files. A value of 0 indicates no byte limit. It is enforced together
        with ``threshold``: when either limit is exceeded, expired entries
        are removed first, then entries with the oldest expiration
        timestamp, until both limits are satisfied. Management files (the
        entry count and byte ledgers) and uncommitted temporary files are
        never counted towards either limit. A single entry larger than
        ``max_bytes`` makes :meth:`set` fail without destroying any
        previously stored value.

        .. versionadded:: 0.18.0
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
    """

    #: used for temporary files by the FileSystemCache
    _fs_transaction_suffix = ".__wz_cache"
    #: keep amount of files in a cache element
    _fs_count_file = "__wz_cache_count"
    #: keep total size in bytes of the committed cache files
    _fs_bytes_file = "__wz_cache_bytes"

    serializer = FileSystemSerializer()

    def __init__(
        self,
        cache_dir: str,
        threshold: int = 500,
        default_timeout: int | dt.timedelta = 300,
        mode: int | None = None,
        hash_method: _t.Any = hashlib.sha256,
        ignore_delete_many_errors: bool = True,
        max_bytes: int = 0,
    ):
        BaseCache.__init__(
            self, default_timeout, ignore_delete_many_errors=ignore_delete_many_errors
        )
        self._path = cache_dir
        self._threshold = threshold
        self._max_bytes = max_bytes
        self._hash_method = hash_method

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
            self._update_count(value=len(list(self._list_dir())))
        if self._max_bytes != 0:
            # reconcile the byte ledger with the real files on disk
            total_size = 0
            for fname in self._list_dir():
                try:
                    total_size += os.path.getsize(fname)
                except OSError:
                    pass
            self._update_bytes(value=total_size)

    def _get_compatible_platform_mode(self) -> int:
        mode = 0o600  # nix systems
        if platform.system() == "Windows":
            mode = stat.S_IWRITE
        return mode

    @property
    def _file_count(self) -> int:
        return self.get(self._fs_count_file) or 0

    @property
    def _file_bytes(self) -> int:
        """Total size in bytes of the committed cache entry files.

        Only tracked when ``max_bytes`` is set; management files and
        uncommitted temporary files are never included.
        """
        return self.get(self._fs_bytes_file) or 0

    def _update_count(self, delta: int | None = None, value: int | None = None) -> None:
        # If we have no threshold, don't count files
        if self._threshold == 0:
            return
        if delta:
            new_count = self._file_count + delta
        else:
            new_count = value or 0
        self.set(self._fs_count_file, new_count, mgmt_element=True)

    def _update_bytes(
        self, delta: int | None = None, value: int | None = None
    ) -> None:
        # If we have no byte budget, don't track file sizes
        if self._max_bytes == 0:
            return
        if delta is not None:
            new_size = self._file_bytes + delta
        else:
            new_size = value or 0
        self.set(self._fs_bytes_file, new_size, mgmt_element=True)

    def _normalize_timeout(self, timeout: int | dt.timedelta | None) -> int:
        normalized_timeout = BaseCache._normalize_timeout(self, timeout)
        if normalized_timeout != 0:
            normalized_timeout = int(time()) + normalized_timeout
        return int(normalized_timeout)

    def _is_mgmt(self, name: str) -> bool:
        mgmt_names = {
            self._get_filename(mgmt_file).split(os.sep)[-1]
            for mgmt_file in (self._fs_count_file, self._fs_bytes_file)
        }
        return name in mgmt_names or name.endswith(self._fs_transaction_suffix)

    def _list_dir(self) -> _t.Generator[str, None, None]:
        """return a list of (fully qualified) cache filenames"""
        return (
            os.path.join(self._path, fn)
            for fn in os.listdir(self._path)
            if not self._is_mgmt(fn)
        )

    def _over_threshold(self) -> bool:
        return self._threshold != 0 and self._file_count > self._threshold

    def _over_max_bytes(self, extra_bytes: int = 0) -> bool:
        return (
            self._max_bytes != 0
            and self._file_bytes + extra_bytes > self._max_bytes
        )

    def _remove_expired(self, now: float) -> None:
        for fname in self._list_dir():
            try:
                with self._safe_stream_open(fname, "rb") as f:
                    expires = struct.unpack("I", f.read(4))[0]
                    fsize = os.fstat(f.fileno()).st_size
                if expires != 0 and expires < now:
                    os.remove(fname)
                    self._update_count(delta=-1)
                    self._update_bytes(delta=-fsize)
            except FileNotFoundError:
                pass
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )

    def _remove_older(self, extra_bytes: int = 0) -> bool:
        exp_fname_tuples = []
        for fname in self._list_dir():
            try:
                with self._safe_stream_open(fname, "rb") as f:
                    timestamp = struct.unpack("I", f.read(4))[0]
                    fsize = os.fstat(f.fileno()).st_size
                    exp_fname_tuples.append((timestamp, fsize, fname))
            except FileNotFoundError:
                pass
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
        fname_sorted = sorted(exp_fname_tuples, key=lambda item: item[0])
        for _, fsize, fname in fname_sorted:
            try:
                os.remove(fname)
                self._update_count(delta=-1)
                self._update_bytes(delta=-fsize)
            except FileNotFoundError:
                pass
            except OSError:
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
                return False
            if not self._over_threshold() and not self._over_max_bytes(extra_bytes):
                break
        return True

    def _prune(self, extra_bytes: int = 0) -> None:
        if self._over_threshold() or self._over_max_bytes(extra_bytes):
            now = time()
            self._remove_expired(now)
        # if still over the entry count or the byte budget
        if self._over_threshold() or self._over_max_bytes(extra_bytes):
            self._remove_older(extra_bytes)

    def clear(self) -> bool:
        removed_bytes = 0
        for i, fname in enumerate(self._list_dir()):
            try:
                fsize = os.path.getsize(fname)
                os.remove(fname)
                removed_bytes += fsize
            except FileNotFoundError:
                pass
            except OSError:
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
                self._update_count(delta=-i)
                self._update_bytes(delta=-removed_bytes)
                return False
        self._update_count(value=0)
        self._update_bytes(value=0)
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

    def add(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool:
        filename = self._get_filename(key)
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
        normalized_timeout = self._normalize_timeout(timeout)
        filename = self._get_filename(key)
        overwrite = os.path.isfile(filename)
        old_size = 0
        if overwrite and not mgmt_element and self._max_bytes != 0:
            try:
                old_size = os.path.getsize(filename)
            except OSError:
                old_size = 0

        # Serialize into a temporary file first, so that a failure
        # neither destroys the previous value nor leaks disk space.
        try:
            fd, tmp = tempfile.mkstemp(
                suffix=self._fs_transaction_suffix, dir=self._path
            )
        except OSError:
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )
            return False
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(struct.pack("I", normalized_timeout))
                self.serializer.dump(value, f)
            fsize = os.path.getsize(tmp)
        except OSError:
            _remove_quietly(tmp)
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )
            return False
        except BaseException:
            # e.g. a serializer failure: drop the uncommitted temporary
            # file and leave the stored counters untouched
            _remove_quietly(tmp)
            raise

        # Don't prune on management element update, to avoid loop
        if not mgmt_element:
            # A single entry larger than the whole budget can never be
            # stored; reject it before the previous value is destroyed.
            if self._max_bytes != 0 and fsize > self._max_bytes:
                _remove_quietly(tmp)
                return False
            self._prune(extra_bytes=fsize - old_size)
            # Pruning may have evicted the entry being replaced
            if overwrite and not os.path.isfile(filename):
                overwrite = False
                old_size = 0

        try:
            self._run_safely(os.replace, tmp, filename)
        except OSError:
            _remove_quietly(tmp)
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )
            return False
        if os.path.exists(tmp):
            # os.replace never happened (e.g. a PermissionError persisted
            # through all _run_safely retries): the previous value is
            # still in place, drop the uncommitted temporary file.
            _remove_quietly(tmp)
            return False
        try:
            self._run_safely(os.chmod, filename, self._mode)
        except OSError:
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )

        # The entry is committed; bring the stored counters in line with
        # the new on-disk state. Management elements are not counted.
        if not mgmt_element:
            if not overwrite:
                self._update_count(delta=1)
            self._update_bytes(delta=fsize - old_size)
        return fsize > 0  # function should fail if file is empty

    def delete(self, key: str, mgmt_element: bool = False) -> bool:
        filename = self._get_filename(key)
        try:
            fsize = os.path.getsize(filename)
            os.remove(filename)
        except FileNotFoundError:  # if file doesn't exist we consider it deleted
            return True
        except OSError:
            logging.warning("Exception raised while handling cache file", exc_info=True)
            return False
        else:
            # Management elements should not count towards threshold
            if not mgmt_element:
                self._update_count(delta=-1)
                self._update_bytes(delta=-fsize)
            return True

    def has(self, key: str) -> bool:
        filename = self._get_filename(key)
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
