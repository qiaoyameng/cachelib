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
    :param collision_safe: If True, store the original key inside each cache
        file and verify it on every access, so distinct keys that hash to the
        same filename (e.g. when a weak or short ``hash_method`` is used) are
        kept isolated instead of overwriting, mis-hitting or deleting each
        other. Colliding keys are stored in "slot" files named
        ``<hash>``, ``<hash>.1``, ``<hash>.2`` and so on.

        In this mode the internal entry-count bookkeeping is kept in a
        reserved file named ``__wz_cache_count`` (a name no hex digest can
        collide with) instead of a hashed management element, so key hashes
        colliding with the management element cannot corrupt it.

        Files written by the legacy format (``collision_safe=False``) do not
        contain their original key and the hash is one-way, so they cannot be
        migrated or attributed to any key. They are treated as foreign: never
        returned by :meth:`get`/:meth:`has`, never removed by
        :meth:`delete`, and not counted towards the threshold. They are only
        removed by :meth:`clear` (or manually). Enable this mode on an empty
        cache directory, or call :meth:`clear` after enabling it.

        .. versionadded:: 0.18.0
    :param ignore_delete_many_errors: If False, delete_many() will raise
        a RuntimeError if any key fails to delete. Keys that do not
        exist are considered successfully deleted and do not raise.

        .. versionadded:: 0.16.0
    """

    #: used for temporary files by the FileSystemCache
    _fs_transaction_suffix = ".__wz_cache"
    #: keep amount of files in a cache element
    _fs_count_file = "__wz_cache_count"
    #: marks a collision-safe record; followed by the stored original key
    _fs_collision_safe_magic = b"\x89CSK"

    serializer = FileSystemSerializer()

    def __init__(
        self,
        cache_dir: str,
        threshold: int = 500,
        default_timeout: int | dt.timedelta = 300,
        mode: int | None = None,
        hash_method: _t.Any = hashlib.sha256,
        collision_safe: bool = False,
        ignore_delete_many_errors: bool = True,
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
            self._update_count(value=self._entry_count())

    def _entry_count(self) -> int:
        """Count the cache entries currently stored in the cache dir.

        In collision-safe mode a file only counts as an entry if its
        content is a verifiable record carrying an original key; foreign
        files (e.g. legacy format files) are not counted, since their
        ownership cannot be established from the filename alone.
        """
        if not self._collision_safe:
            return len(list(self._list_dir()))
        count = 0
        for fname in self._list_dir():
            try:
                with self._safe_stream_open(fname, "rb") as f:
                    _, stored_key = self._read_record_header(f)
            except (OSError, EOFError, struct.error):
                continue
            if stored_key is not None:
                count += 1
        return count

    def _get_compatible_platform_mode(self) -> int:
        mode = 0o600  # nix systems
        if platform.system() == "Windows":
            mode = stat.S_IWRITE
        return mode

    @property
    def _file_count(self) -> int:
        if self._collision_safe:
            return self._read_count_file()
        return self.get(self._fs_count_file) or 0

    def _update_count(self, delta: int | None = None, value: int | None = None) -> None:
        # If we have no threshold, don't count files
        if self._threshold == 0:
            return
        if delta:
            new_count = self._file_count + delta
        else:
            new_count = value or 0
        if self._collision_safe:
            self._write_count_file(new_count)
        else:
            self.set(self._fs_count_file, new_count, mgmt_element=True)

    def _count_filename(self) -> str:
        # The literal name cannot collide with any key hash, since hex
        # digests never contain underscores.
        return os.path.join(self._path, self._fs_count_file)

    def _read_count_file(self) -> int:
        try:
            with self._safe_stream_open(self._count_filename(), "rb") as f:
                struct.unpack("I", f.read(4))[0]
                return self.serializer.load(f) or 0
        except (OSError, EOFError, struct.error):
            return 0

    def _write_count_file(self, new_count: int) -> None:
        filename = self._count_filename()
        try:
            fd, tmp = tempfile.mkstemp(
                suffix=self._fs_transaction_suffix, dir=self._path
            )
            with os.fdopen(fd, "wb") as f:
                f.write(struct.pack("I", 0))  # never expires
                self.serializer.dump(new_count, f)
            self._run_safely(os.replace, tmp, filename)
            self._run_safely(os.chmod, filename, self._mode)
        except OSError:
            logging.warning(
                "Exception raised while handling cache file '%s'",
                filename,
                exc_info=True,
            )

    def _normalize_timeout(self, timeout: int | dt.timedelta | None) -> int:
        normalized_timeout = BaseCache._normalize_timeout(self, timeout)
        if normalized_timeout != 0:
            normalized_timeout = int(time()) + normalized_timeout
        return int(normalized_timeout)

    def _is_mgmt(self, name: str) -> bool:
        if self._collision_safe:
            # the count file uses a reserved literal name in this mode
            return name == self._fs_count_file or name.endswith(
                self._fs_transaction_suffix
            )
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

    def _remove_expired(self, now: float) -> None:
        for fname in self._list_dir():
            try:
                with self._safe_stream_open(fname, "rb") as f:
                    if self._collision_safe:
                        expires, stored_key = self._read_record_header(f)
                        # foreign files carry no key; leave them alone
                        if stored_key is None:
                            continue
                    else:
                        expires = struct.unpack("I", f.read(4))[0]
                if expires != 0 and expires < now:
                    os.remove(fname)
                    self._update_count(delta=-1)
            except FileNotFoundError:
                pass
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )

    def _remove_older(self) -> bool:
        exp_fname_tuples = []
        for fname in self._list_dir():
            try:
                with self._safe_stream_open(fname, "rb") as f:
                    if self._collision_safe:
                        timestamp, stored_key = self._read_record_header(f)
                        # foreign files carry no key; leave them alone
                        if stored_key is None:
                            continue
                    else:
                        timestamp = struct.unpack("I", f.read(4))[0]
                    exp_fname_tuples.append((timestamp, fname))
            except FileNotFoundError:
                pass
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
        fname_sorted = (
            fname for _, fname in sorted(exp_fname_tuples, key=lambda item: item[0])
        )
        for fname in fname_sorted:
            try:
                os.remove(fname)
                self._update_count(delta=-1)
            except FileNotFoundError:
                pass
            except OSError:
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
                return False
            if not self._over_threshold():
                break
        return True

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

    def _get_hash(self, key: str | _t.Any) -> str:
        return self._get_filename(key).split(os.sep)[-1]

    def _slot_filename(self, bkey_hash: str, slot: int) -> str:
        name = bkey_hash if slot == 0 else f"{bkey_hash}.{slot}"
        return os.path.join(self._path, name)

    def _slot_files(self, bkey_hash: str) -> dict[int, str]:
        """Map of slot index -> filename for existing slots of a hash."""
        slots = {}
        prefix = bkey_hash + "."
        for fn in os.listdir(self._path):
            if fn == bkey_hash:
                slots[0] = os.path.join(self._path, fn)
            elif fn.startswith(prefix) and fn[len(prefix) :].isdigit():
                slots[int(fn[len(prefix) :])] = os.path.join(self._path, fn)
        return slots

    def _read_record_header(self, f: _t.IO[bytes]) -> tuple[int, str | None]:
        """Read the expiry timestamp and stored key of a collision-safe
        record. Returns ``(expires, None)`` for files that do not carry a
        verifiable key (legacy format or foreign files); their ownership
        must not be guessed.
        """
        expires = struct.unpack("I", f.read(4))[0]
        if f.read(4) != self._fs_collision_safe_magic:
            return expires, None
        key_len = struct.unpack("I", f.read(4))[0]
        try:
            stored_key: str | None = f.read(key_len).decode("utf-8")
        except UnicodeDecodeError:
            stored_key = None
        return expires, stored_key

    @contextmanager
    def _open_record(
        self, key: str
    ) -> _t.Generator[tuple[str | None, int, _t.IO[bytes] | None], None, None]:
        """Yield ``(filename, expires, file)`` for the collision-safe record
        storing ``key``, with the file positioned at the serialized payload.
        Yields ``(None, 0, None)`` if no verifiable record stores ``key``.
        """
        slots = self._slot_files(self._get_hash(key))
        for slot in sorted(slots):
            fname = slots[slot]
            try:
                with self._safe_stream_open(fname, "rb") as f:
                    expires, stored_key = self._read_record_header(f)
                    if stored_key == key:
                        yield fname, expires, f
                        return
            except FileNotFoundError:
                continue
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
        yield None, 0, None

    def _get_collision_safe(self, key: str) -> _t.Any:
        with self._open_record(key) as (fname, expires, f):
            if fname is None or f is None:
                return None
            if expires != 0 and expires < time():
                return None
            try:
                return self.serializer.load(f)
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
                return None

    def _set_collision_safe(
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
        bkey_hash = self._get_hash(key)
        slots = self._slot_files(bkey_hash)

        # find the slot already storing this exact key, if any
        filename = None
        overwrite = False
        for slot in sorted(slots):
            fname = slots[slot]
            try:
                with self._safe_stream_open(fname, "rb") as f:
                    _, stored_key = self._read_record_header(f)
            except FileNotFoundError:
                continue
            except (OSError, EOFError, struct.error):
                logging.warning(
                    "Exception raised while handling cache file '%s'",
                    fname,
                    exc_info=True,
                )
                continue
            if stored_key == key:
                filename = fname
                overwrite = True
                break

        # otherwise claim the first free slot; foreign files occupying a
        # slot are left untouched since their ownership is unknown
        if filename is None:
            slot = 0
            while slot in slots:
                slot += 1
            filename = self._slot_filename(bkey_hash, slot)

        try:
            fd, tmp = tempfile.mkstemp(
                suffix=self._fs_transaction_suffix, dir=self._path
            )
            with os.fdopen(fd, "wb") as f:
                f.write(struct.pack("I", normalized_timeout))
                f.write(self._fs_collision_safe_magic)
                bkey = key.encode("utf-8")
                f.write(struct.pack("I", len(bkey)))
                f.write(bkey)
                self.serializer.dump(value, f)

            self._run_safely(os.replace, tmp, filename)
            self._run_safely(os.chmod, filename, self._mode)

            fsize = Path(filename).stat().st_size
        except OSError:
            # the previous complete record, if any, is left in place
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

    def _delete_collision_safe(self, key: str, mgmt_element: bool = False) -> bool:
        with self._open_record(key) as (fname, _, _f):
            filename = fname
        if filename is None:
            # if no verifiable record stores the key we consider it deleted
            return True
        try:
            os.remove(filename)
        except FileNotFoundError:
            return True
        except OSError:
            logging.warning("Exception raised while handling cache file", exc_info=True)
            return False
        else:
            # Management elements should not count towards threshold
            if not mgmt_element:
                self._update_count(delta=-1)
            return True

    def _has_collision_safe(self, key: str) -> bool:
        with self._open_record(key) as (fname, expires, _f):
            if fname is None:
                return False
            return expires == 0 or expires >= time()

    def get(self, key: str) -> _t.Any:
        if self._collision_safe:
            return self._get_collision_safe(key)
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
        if self._collision_safe:
            with self._open_record(key) as (fname, _, _f):
                if fname is not None:
                    return False
            return self.set(key, value, timeout)
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
        if self._collision_safe:
            return self._set_collision_safe(
                key, value, timeout, mgmt_element=mgmt_element
            )
        # Management elements have no timeout
        if mgmt_element:
            timeout = 0
        # Don't prune on management element update, to avoid loop
        else:
            self._prune()

        normalized_timeout = self._normalize_timeout(timeout)
        filename = self._get_filename(key)
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

    def delete(self, key: str, mgmt_element: bool = False) -> bool:
        if self._collision_safe:
            return self._delete_collision_safe(key, mgmt_element=mgmt_element)
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

    def has(self, key: str) -> bool:
        if self._collision_safe:
            return self._has_collision_safe(key)
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
