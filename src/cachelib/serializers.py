import hmac
import io
import json
import logging
import pickle
import struct
import typing as _t
from hashlib import sha256


_AUTH_FORMAT_VERSION = 1
_AUTH_HEADER = struct.Struct(">BBI")
_AUTH_SIGNATURE_SIZE = sha256().digest_size
_AUTH_DEFAULT_MAX_PAYLOAD_SIZE = 10 * 1024 * 1024


class _InvalidEnvelope(ValueError):
    """An authenticated value could not be safely parsed."""


def _key_id_bytes(key_id: str | bytes | bytearray) -> bytes:
    if isinstance(key_id, str):
        value = key_id.encode("utf-8")
    elif isinstance(key_id, (bytes, bytearray)):
        value = bytes(key_id)
    else:
        raise TypeError("key id must be str or bytes")
    if not value or len(value) > 255:
        raise ValueError("key id must contain between 1 and 255 bytes")
    return value


def _secret_bytes(secret: bytes | bytearray) -> bytes:
    if not isinstance(secret, (bytes, bytearray)):
        raise TypeError("HMAC secret must be bytes")
    value = bytes(secret)
    if not value:
        raise ValueError("HMAC secret must not be empty")
    return value


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


class AuthenticatedSerializer(BaseSerializer):
    """Authenticate serialized values with HMAC-SHA256.

    The wrapped serializer is only called after the complete envelope has
    been parsed and its signature has been verified. New values are signed
    with ``secret``. Values signed with a key in ``old_keys`` can be read,
    but those keys are never used when writing.

    The envelope format is version ``1`` and consists of a six-byte header
    (version, key-id length, payload length), the key id, the inner payload,
    and a 32-byte signature. The signature covers the header, key id, and
    payload. Unauthenticated values from older releases are rejected.

    :param serializer: serializer used for the authenticated payload.
    :param secret: HMAC key used for new values.
    :param key_id: identifier written with new values.
    :param old_keys: mapping of explicitly accepted previous key ids to
        HMAC keys.
    :param max_payload_size: maximum accepted inner payload size in bytes.
    """

    def __init__(
        self,
        serializer: BaseSerializer,
        secret: bytes | bytearray,
        key_id: str | bytes | bytearray = "current",
        old_keys: _t.Mapping[str | bytes | bytearray, bytes | bytearray]
        | None = None,
        max_payload_size: int = _AUTH_DEFAULT_MAX_PAYLOAD_SIZE,
    ) -> None:
        if not isinstance(max_payload_size, int) or isinstance(
            max_payload_size, bool
        ):
            raise TypeError("max_payload_size must be an int")
        if max_payload_size <= 0 or max_payload_size > 0xFFFFFFFF:
            raise ValueError("max_payload_size must be between 1 and 4294967295 bytes")

        current_key_id = _key_id_bytes(key_id)
        self._current_key_id = current_key_id
        self._current_secret = _secret_bytes(secret)
        self._serializer = serializer
        self._max_payload_size = max_payload_size
        self._keys = {current_key_id: self._current_secret}

        if old_keys is not None:
            for old_key_id, old_secret in old_keys.items():
                old_key_id_bytes = _key_id_bytes(old_key_id)
                if old_key_id_bytes in self._keys:
                    raise ValueError("duplicate key id")
                self._keys[old_key_id_bytes] = _secret_bytes(old_secret)

    @classmethod
    def from_keys(
        cls,
        serializer: BaseSerializer,
        keys: _t.Mapping[str | bytes | bytearray, bytes | bytearray],
        current_key: str | bytes | bytearray,
        max_payload_size: int = _AUTH_DEFAULT_MAX_PAYLOAD_SIZE,
    ) -> "AuthenticatedSerializer":
        """Create a serializer from a complete key mapping."""
        current_key_id = _key_id_bytes(current_key)
        normalized_keys = {}
        for key_id, secret in keys.items():
            normalized_key_id = _key_id_bytes(key_id)
            if normalized_key_id in normalized_keys:
                raise ValueError("duplicate key id")
            normalized_keys[normalized_key_id] = _secret_bytes(secret)
        try:
            secret = normalized_keys[current_key_id]
        except KeyError:
            raise ValueError("current key id is not configured") from None
        old_keys = {
            key_id: old_secret
            for key_id, old_secret in normalized_keys.items()
            if key_id != current_key_id
        }
        return cls(
            serializer,
            secret,
            current_key_id,
            old_keys=old_keys,
            max_payload_size=max_payload_size,
        )

    def _warn(self, e: Exception) -> None:
        logging.warning("Refusing unauthenticated cache value: %s", e)

    def _signature(self, signed_data: bytes, secret: bytes | bytearray) -> bytes:
        return hmac.new(secret, signed_data, sha256).digest()

    def _envelope(self, payload: bytes | bytearray) -> bytes:
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("serialized payload must be bytes")
        if len(payload) > self._max_payload_size:
            raise _InvalidEnvelope("payload exceeds configured size limit")

        header = _AUTH_HEADER.pack(
            _AUTH_FORMAT_VERSION,
            len(self._current_key_id),
            len(payload),
        )
        signed_data = header + self._current_key_id + bytes(payload)
        return signed_data + self._signature(signed_data, self._current_secret)

    def _dump_bytes(self, value: _t.Any, protocol: int | None) -> bytes | None:
        if protocol is None:
            payload = self._serializer.dumps(value)
        else:
            payload = self._serializer.dumps(value, protocol)
        if payload is None:
            return None
        try:
            return self._envelope(payload)
        except (TypeError, _InvalidEnvelope) as e:
            self._warn(e)
            return None

    def dumps(self, value: _t.Any, protocol: int | None = None) -> bytes | None:
        return self._dump_bytes(value, protocol)

    def dump(
        self,
        value: _t.Any,
        f: _t.IO[bytes],
        protocol: int | None = None,
    ) -> None:
        payload = self._dump_bytes(value, protocol)
        if payload is not None:
            f.write(payload)

    def _read_exact(self, f: _t.BinaryIO, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = f.read(min(remaining, 65536))
            if not chunk:
                raise _InvalidEnvelope("truncated envelope")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _load_envelope_stream(self, f: _t.BinaryIO) -> bytes:
        header = f.read(_AUTH_HEADER.size)
        if len(header) < _AUTH_HEADER.size:
            raise _InvalidEnvelope("truncated envelope")

        version, key_id_length, payload_length = _AUTH_HEADER.unpack(header)
        if version != _AUTH_FORMAT_VERSION:
            raise _InvalidEnvelope("unknown format version")
        if key_id_length == 0:
            raise _InvalidEnvelope("invalid key id")
        if payload_length == 0:
            raise _InvalidEnvelope("empty payload")
        if payload_length > self._max_payload_size:
            raise _InvalidEnvelope("payload exceeds configured size limit")

        key_id = self._read_exact(f, key_id_length)
        secret = self._keys.get(key_id)
        if secret is None:
            raise _InvalidEnvelope("unknown key id")

        payload = self._read_exact(f, payload_length)
        signature = self._read_exact(f, _AUTH_SIGNATURE_SIZE)
        if f.read(1):
            raise _InvalidEnvelope("unexpected trailing data")

        signed_data = header + key_id + payload
        expected_signature = self._signature(signed_data, secret)
        if not hmac.compare_digest(signature, expected_signature):
            raise _InvalidEnvelope("invalid signature")
        return payload

    def loads(self, bvalue: bytes | bytearray | None) -> _t.Any:
        if bvalue is None:
            return None
        try:
            payload = self._load_envelope_stream(io.BytesIO(bvalue))
        except (TypeError, struct.error, _InvalidEnvelope) as e:
            self._warn(e)
            return None
        return self._serializer.loads(payload)

    def load(self, f: _t.BinaryIO) -> _t.Any:
        try:
            payload = self._load_envelope_stream(f)
        except (TypeError, struct.error, _InvalidEnvelope) as e:
            self._warn(e)
            return None
        return self._serializer.load(io.BytesIO(payload))


HMACSerializer = AuthenticatedSerializer


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
