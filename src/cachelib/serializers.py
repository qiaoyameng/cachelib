from __future__ import annotations

import hashlib
import hmac
import json
import logging
import pickle
import struct
import typing as _t
from collections.abc import Mapping


_FORMAT_VERSION = 1
_DIGEST_SIZE = hashlib.sha256().digest_size
_HEADER_SIZE = 10


class _MalformedAuthenticatedValue(ValueError):
    pass


class AuthenticationError(ValueError):
    """Raised when an authenticated cache value cannot be verified."""


def _as_bytes(value: str | bytes | bytearray, name: str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"{name} must be bytes or a string")


class HMACSerializer:
    """Authenticate serialized values with HMAC-SHA256.

    The wrapped serializer is only called after the complete value has been
    parsed and authenticated. New values are signed with the current key.
    Keys in ``old_keys`` are accepted on reads but never used for writes.

    The version 1 binary format is::

        version (1 byte) || key-id length (1 byte) ||
        payload length (8 bytes, big endian) || key id || payload ||
        HMAC-SHA256

    The HMAC covers the version, key id, payload length, key id, and
    payload. Values without this envelope are rejected.

    :param serializer: serializer used for the authenticated payload.
    :param current_key: key used to sign new values.
    :param current_key_id: identifier stored in newly written values.
    :param old_keys: mapping of explicitly accepted legacy key identifiers
        to keys.
    :param max_payload_size: maximum accepted payload size in bytes.
    """

    def __init__(
        self,
        serializer: BaseSerializer | None = None,
        current_key: str | bytes | bytearray | None = None,
        *,
        current_key_id: str | bytes | bytearray = b"current",
        old_keys: Mapping[str | bytes | bytearray, str | bytes | bytearray]
        | None = None,
        max_payload_size: int = 10 * 1024 * 1024,
    ) -> None:
        self.serializer = serializer if serializer is not None else BaseSerializer()
        self._current_key_id = _as_bytes(current_key_id, "current_key_id")
        if not self._current_key_id or len(self._current_key_id) > 255:
            raise ValueError("current_key_id must contain between 1 and 255 bytes")

        if current_key is None:
            raise TypeError("current_key is required")
        self._current_key = _as_bytes(current_key, "current_key")
        if not self._current_key:
            raise ValueError("current_key must not be empty")

        if not isinstance(max_payload_size, int) or isinstance(max_payload_size, bool):
            raise TypeError("max_payload_size must be an integer")
        if max_payload_size <= 0 or max_payload_size > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("max_payload_size must be a positive 64-bit integer")
        self.max_payload_size = max_payload_size

        self._keys: dict[bytes, bytes] = {}
        if old_keys is not None:
            for key_id, key in old_keys.items():
                normalized_key_id = _as_bytes(key_id, "old key id")
                if not normalized_key_id or len(normalized_key_id) > 255:
                    raise ValueError(
                        "old key ids must contain between 1 and 255 bytes"
                    )
                if normalized_key_id in self._keys:
                    raise ValueError("old_keys contains a duplicate key id")
                normalized_key = _as_bytes(key, "old key")
                if not normalized_key:
                    raise ValueError("old keys must not be empty")
                self._keys[normalized_key_id] = normalized_key

        if self._current_key_id in self._keys:
            raise ValueError("current_key_id must not also be an old key id")
        self._keys[self._current_key_id] = self._current_key

    def _reject(self, reason: str) -> _t.NoReturn:
        logging.warning("Rejected an unauthenticated cache value")
        if reason == "malformed":
            raise _MalformedAuthenticatedValue(reason)
        raise AuthenticationError(reason)

    def _authenticate(
        self,
        version: int,
        key_id: bytes,
        payload_length: int,
        payload: bytes,
        signature: bytes,
    ) -> bytes:
        if version != _FORMAT_VERSION:
            self._reject("unknown version")
        key = self._keys.get(key_id)
        if key is None:
            self._reject("unknown key")

        signed = (
            struct.pack("!BBQ", version, len(key_id), payload_length)
            + key_id
            + payload
        )
        expected = hmac.digest(key, signed, "sha256")
        if not hmac.compare_digest(signature, expected):
            self._reject("invalid signature")
        return payload

    def _pack(self, payload: bytes) -> bytes:
        if len(payload) > self.max_payload_size:
            raise ValueError("serialized payload exceeds max_payload_size")
        header = struct.pack(
            "!BBQ", _FORMAT_VERSION, len(self._current_key_id), len(payload)
        )
        signed = header + self._current_key_id + payload
        signature = hmac.digest(self._current_key, signed, "sha256")
        return signed + signature

    def _parse_bytes(self, value: bytes) -> bytes:
        if not isinstance(value, bytes):
            self._reject("malformed")
        if len(value) < _HEADER_SIZE + _DIGEST_SIZE:
            self._reject("malformed")

        version, key_id_length, payload_length = struct.unpack(
            "!BBQ", value[:_HEADER_SIZE]
        )
        if version != _FORMAT_VERSION:
            self._reject("unknown version")
        if key_id_length == 0:
            self._reject("malformed")
        if payload_length > self.max_payload_size:
            self._reject("payload too large")

        payload_offset = _HEADER_SIZE + key_id_length
        signature_offset = payload_offset + payload_length
        end = signature_offset + _DIGEST_SIZE
        if end != len(value):
            self._reject("malformed")

        key_id = value[_HEADER_SIZE:payload_offset]
        payload = value[payload_offset:signature_offset]
        signature = value[signature_offset:end]
        return self._authenticate(
            version,
            key_id,
            payload_length,
            payload,
            signature,
        )

    def _read_exact(self, stream: _t.BinaryIO, size: int) -> bytes:
        data = stream.read(size)
        if not isinstance(data, bytes) or len(data) != size:
            self._reject("malformed")
        return data

    def _parse_stream(self, stream: _t.BinaryIO) -> bytes:
        header = self._read_exact(stream, _HEADER_SIZE)
        version, key_id_length, payload_length = struct.unpack("!BBQ", header)
        if version != _FORMAT_VERSION:
            self._reject("unknown version")
        if key_id_length == 0:
            self._reject("malformed")
        if payload_length > self.max_payload_size:
            self._reject("payload too large")

        key_id = self._read_exact(stream, key_id_length)
        payload = self._read_exact(stream, payload_length)
        signature = self._read_exact(stream, _DIGEST_SIZE)
        if stream.read(1):
            self._reject("malformed")
        return self._authenticate(
            version, key_id, payload_length, payload, signature
        )

    def dump(
        self, value: _t.Any, f: _t.IO[bytes], protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> None:
        payload = self.serializer.dumps(value, protocol)
        if payload is None:
            return
        if not isinstance(payload, bytes):
            raise TypeError("serializer.dumps must return bytes")
        f.write(self._pack(payload))

    def load(self, f: _t.BinaryIO) -> _t.Any:
        try:
            payload = self._parse_stream(f)
        except (
            _MalformedAuthenticatedValue,
            AuthenticationError,
            OSError,
            struct.error,
        ):
            return None
        return self.serializer.loads(payload)

    def dumps(
        self, value: _t.Any, protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> bytes | None:
        payload = self.serializer.dumps(value, protocol)
        if payload is None:
            return None
        if not isinstance(payload, bytes):
            raise TypeError("serializer.dumps must return bytes")
        return self._pack(payload)

    def loads(self, bvalue: bytes) -> _t.Any:
        try:
            if isinstance(bvalue, bytearray):
                bvalue = bytes(bvalue)
            elif isinstance(bvalue, memoryview):
                bvalue = bvalue.tobytes()
            payload = self._parse_bytes(bvalue)
        except (_MalformedAuthenticatedValue, AuthenticationError, struct.error):
            return None
        return self.serializer.loads(payload)


class BaseSerializer:
    """This is the base interface for all default serializers.

    BaseSerializer.load and BaseSerializer.dump will
    default to pickle.load and pickle.dump. This is currently
    used only by FileSystemCache which dumps/loads to/from a file stream.
    """

    def _warn(self, e: Exception) -> None:
        logging.warning(f"An exception has been raised during an operation: {e}")

    def dump(
        self, value: int, f: _t.IO[bytes], protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> None:
        try:
            pickle.dump(value, f, protocol)
        except (pickle.PickleError, pickle.PicklingError) as e:
            self._warn(e)

    def load(self, f: _t.BinaryIO) -> _t.Any:
        try:
            data = pickle.load(f)
        except pickle.PickleError as e:
            self._warn(e)
            return None
        else:
            return data

    """BaseSerializer.loads and BaseSerializer.dumps
    work on top of pickle.loads and pickle.dumps. Dumping/loading
    strings and byte strings is the default for most cache types.
    """

    def dumps(
        self, value: _t.Any, protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> bytes | None:
        try:
            serialized = pickle.dumps(value, protocol)
        except (pickle.PickleError, pickle.PicklingError) as e:
            self._warn(e)
            return None
        return serialized

    def loads(self, bvalue: bytes) -> _t.Any:
        try:
            data = pickle.loads(bvalue)
        except pickle.PickleError as e:
            self._warn(e)
            return None
        else:
            return data


AuthenticatedSerializer = HMACSerializer


class BaseRedisSerializer(BaseSerializer):
    """Base serializer for Redis compatible caches."""

    def dumps(self, value: _t.Any, protocol: int = pickle.HIGHEST_PROTOCOL) -> bytes:
        """Dumps an object into a string for redis. Integers are stored as
        plain ASCII so they remain compatible with the native
        ``INCRBY``/``DECRBY`` operations used by :meth:`inc`/:meth:`dec`;
        everything else is pickled.
        """
        # check type instead of isinstance since bool is an int subclass
        if type(value) is int:
            return str(value).encode("ascii")
        return b"!" + pickle.dumps(value, protocol)

    def loads(self, value: bytes | None) -> _t.Any:
        """The reversal of :meth:`dumps`. This might be called with
        None.
        """
        if value is None:
            return None
        if value.startswith(b"!"):
            try:
                return pickle.loads(value[1:])
            except pickle.PickleError:
                return None
        try:
            return int(value)
        except ValueError:
            # before 0.8 we did not have serialization. Still support that.
            return value


class JSONSerializer(BaseSerializer):
    """Generic JSON serializer for caches.

    Use this serializer if you want to serialize to JSON instead of pickle.
    Note that JSON does not support serialization of Python bytes objects.
    If you need to cache bytes, use the default pickle-based serializer.
    This serializer is not used by default.
    """

    def dump(
        self, value: _t.Any, f: _t.IO[bytes], *args: _t.Any, **kwargs: _t.Any
    ) -> None:
        try:
            f.write(json.dumps(value, *args, **kwargs).encode())
        except (TypeError, ValueError) as e:
            self._warn(e)

    def load(self, f: _t.IO[bytes], *args: _t.Any, **kwargs: _t.Any) -> _t.Any:
        try:
            data = json.loads(f.read(), *args, **kwargs)
        except (TypeError, ValueError) as e:
            self._warn(e)
            return None
        else:
            return data

    def dumps(self, value: _t.Any, *args: _t.Any, **kwargs: _t.Any) -> bytes | None:
        try:
            serialized = json.dumps(value, *args, **kwargs).encode()
        except (TypeError, ValueError) as e:
            self._warn(e)
            return None
        return serialized

    def loads(
        self, bvalue: str | bytes | bytearray, *args: _t.Any, **kwargs: _t.Any
    ) -> _t.Any:
        try:
            data = json.loads(bvalue, *args, **kwargs)
        except (TypeError, ValueError) as e:
            self._warn(e)
            return None
        else:
            return data


"""Default serializers for each cache type.

The following classes can be used to further customize
serialiation behaviour. Alternatively, any serializer can be
overridden in order to use a custom serializer with a different
strategy altogether.
"""


class UWSGISerializer(BaseSerializer):
    """Default serializer for UWSGICache."""


class SimpleSerializer(BaseSerializer):
    """Default serializer for SimpleCache."""


class FileSystemSerializer(BaseSerializer):
    """Default serializer for FileSystemCache."""


class MongoDbSerializer(BaseSerializer):
    """Default serializer for MongoDbCache."""


class RedisSerializer(BaseRedisSerializer):
    """Default serializer for RedisCache."""


class ValkeySerializer(BaseRedisSerializer):
    """Default serializer for ValkeyCache."""


class DynamoDbSerializer(BaseRedisSerializer):
    """Default serializer for DynamoDbCache."""
