import hmac
import io
import logging
import pickle
import struct
from hashlib import sha256

import pytest

from cachelib.serializers import AuthenticatedSerializer
from cachelib.serializers import BaseRedisSerializer
from cachelib.serializers import BaseSerializer
from cachelib.serializers import HMACSerializer
from cachelib.serializers import JSONSerializer
from cachelib.serializers import RedisSerializer
from cachelib.serializers import ValkeySerializer

PICKLE_VALUES = [
    None,
    True,
    False,
    0,
    42,
    -1,
    3.14,
    "",
    "hello",
    b"raw bytes",
    [],
    [1, 2, 3],
    (1, 2),
    {},
    {"a": 1, "b": [True, None]},
]

JSON_VALUES = [v for v in PICKLE_VALUES if not isinstance(v, (bytes, tuple))]


class _Unpicklable:
    """Helper that always raises PicklingError"""

    def __reduce__(self):
        raise pickle.PicklingError("cannot pickle")


class CountingSerializer(BaseSerializer):
    def __init__(self):
        self.loads_count = 0
        self.load_count = 0

    def dumps(self, value, protocol=pickle.HIGHEST_PROTOCOL):
        return pickle.dumps(value, protocol)

    def loads(self, bvalue):
        self.loads_count += 1
        return pickle.loads(bvalue)

    def load(self, f):
        self.load_count += 1
        return pickle.load(f)


class TestBaseSerializer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.s = BaseSerializer()

    @pytest.mark.parametrize("value", PICKLE_VALUES)
    def test_dumps_loads_roundtrip(self, value):
        assert self.s.loads(self.s.dumps(value)) == value

    @pytest.mark.parametrize("value", PICKLE_VALUES)
    def test_dump_load_roundtrip(self, value):
        buf = io.BytesIO()
        self.s.dump(value, buf)
        buf.seek(0)
        assert self.s.load(buf) == value

    def test_dumps_returns_none_on_pickle_error(self):
        assert self.s.dumps(_Unpicklable()) is None

    def test_loads_returns_none_on_corrupt_data(self):
        assert self.s.loads(b"not valid pickle data") is None

    def test_load_returns_none_on_corrupt_stream(self):
        assert self.s.load(io.BytesIO(b"not valid pickle data")) is None


class TestBaseRedisSerializer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.s = BaseRedisSerializer()

    @pytest.mark.parametrize("value", PICKLE_VALUES)
    def test_dumps_loads_roundtrip(self, value):
        assert self.s.loads(self.s.dumps(value)) == value

    def test_dumps_adds_bang_prefix(self):
        assert self.s.dumps("hello").startswith(b"!")

    def test_loads_none_returns_none(self):
        assert self.s.loads(None) is None

    @pytest.mark.parametrize("n", [0, 1, 42, -5])
    def test_loads_integer_string_returns_int(self, n):
        # Redis stores counters as plain integers without prefix
        assert self.s.loads(str(n).encode()) == n

    def test_loads_legacy_bytes_returns_raw(self):
        # Before 0.8, values were stored without the "!" prefix
        raw = b"legacy data without prefix"
        assert self.s.loads(raw) == raw

    def test_loads_corrupt_prefixed_data_returns_none(self):
        assert self.s.loads(b"!" + b"corrupt pickle") is None


@pytest.mark.parametrize("cls", [RedisSerializer, ValkeySerializer])
class TestRedisCompatibleSerializers:
    @pytest.mark.parametrize("value", PICKLE_VALUES)
    def test_roundtrip(self, cls, value):
        s = cls()
        assert s.loads(s.dumps(value)) == value


class TestJSONSerializer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.s = JSONSerializer()

    @pytest.mark.parametrize("value", JSON_VALUES)
    def test_dumps_loads_roundtrip(self, value):
        assert self.s.loads(self.s.dumps(value)) == value

    @pytest.mark.parametrize("value", JSON_VALUES)
    def test_dump_load_roundtrip(self, value):
        buf = io.BytesIO()
        self.s.dump(value, buf)
        buf.seek(0)
        assert self.s.load(buf) == value

    def test_dumps_returns_bytes(self):
        assert isinstance(self.s.dumps({"key": "value"}), bytes)

    def test_dump_writes_bytes_to_stream(self):
        buf = io.BytesIO()
        self.s.dump({"key": "value"}, buf)
        assert isinstance(buf.getvalue(), bytes)
        assert buf.getvalue()

    @pytest.mark.parametrize("value", [b"raw bytes", _Unpicklable()])
    def test_dumps_non_serializable_returns_none(self, value):
        assert self.s.dumps(value) is None

    @pytest.mark.parametrize(
        "raw",
        [b'"hello"', '"hello"', bytearray(b'"hello"')],
        ids=["bytes", "str", "bytearray"],
    )
    def test_loads_accepts_bytes_str_bytearray(self, raw):
        assert self.s.loads(raw) == "hello"

    def test_loads_invalid_json_returns_none(self):
        assert self.s.loads(b"not valid json {{{{") is None

    def test_load_invalid_stream_returns_none(self):
        assert self.s.load(io.BytesIO(b"not valid json {{{{")) is None


class TestAuthenticatedSerializer:
    current_secret = b"current-secret"
    old_secret = b"old-secret"

    def make(self, inner=None, **kwargs):
        return AuthenticatedSerializer(
            inner or CountingSerializer(),
            self.current_secret,
            key_id=b"new",
            old_keys={b"old": self.old_secret},
            **kwargs,
        )

    def envelope(self, serializer, value, key_id, secret, version=1):
        payload = pickle.dumps(value)
        header = struct.pack(">BBI", version, len(key_id), len(payload))
        signed_data = header + key_id + payload
        return signed_data + hmac.new(secret, signed_data, sha256).digest()

    def test_roundtrip_uses_current_key(self):
        inner = CountingSerializer()
        serializer = self.make(inner)

        assert serializer.loads(serializer.dumps({"ok": True})) == {"ok": True}
        assert inner.loads_count == 1

    def test_stream_roundtrip_uses_current_key(self):
        inner = CountingSerializer()
        serializer = self.make(inner)
        buf = io.BytesIO()
        serializer.dump({"ok": True}, buf)
        buf.seek(0)

        assert serializer.load(buf) == {"ok": True}
        assert inner.load_count == 1

    def test_stream_roundtrip_with_json_serializer(self):
        serializer = AuthenticatedSerializer(
            JSONSerializer(), self.current_secret
        )
        buf = io.BytesIO()
        serializer.dump({"ok": True}, buf)
        buf.seek(0)

        assert serializer.load(buf) == {"ok": True}

    def test_bytes_roundtrip_with_json_serializer(self):
        serializer = AuthenticatedSerializer(
            JSONSerializer(), self.current_secret
        )

        assert serializer.loads(serializer.dumps({"ok": True})) == {"ok": True}

    @pytest.mark.parametrize(
        "tamper",
        [
            lambda value: value[:0] + bytes([value[0] ^ 1]) + value[1:],
            lambda value: value[:9] + bytes([value[9] ^ 1]) + value[10:],
            lambda value: value[:-1] + bytes([value[-1] ^ 1]),
        ],
        ids=["header", "payload", "signature"],
    )
    def test_tampered_values_do_not_reach_inner_serializer(self, tamper, caplog):
        inner = CountingSerializer()
        serializer = self.make(inner)
        value = serializer.dumps({"ok": True})

        caplog.set_level(logging.WARNING)
        assert serializer.loads(tamper(value)) is None
        assert inner.loads_count == 0
        assert self.current_secret not in caplog.text.encode()
        assert self.old_secret not in caplog.text.encode()

    def test_tampered_stream_values_do_not_reach_inner_serializer(self):
        inner = CountingSerializer()
        serializer = self.make(inner)
        value = serializer.dumps({"ok": True})
        tampered = value[:-1] + bytes([value[-1] ^ 1])

        assert serializer.load(io.BytesIO(tampered)) is None
        assert inner.load_count == 0

    def test_old_key_loads_when_explicitly_configured(self):
        inner = CountingSerializer()
        serializer = self.make(inner)
        value = self.envelope(
            serializer, {"rotated": True}, b"old", self.old_secret
        )

        assert serializer.loads(value) == {"rotated": True}
        assert inner.loads_count == 1

    def test_old_key_is_not_used_for_new_values(self):
        serializer = self.make()
        value = serializer.dumps({"ok": True})

        assert value[6:9] == b"new"

    def test_old_key_rejected_without_configuration(self):
        inner = CountingSerializer()
        serializer = AuthenticatedSerializer(
            inner, self.current_secret, key_id=b"new"
        )
        value = self.envelope(
            serializer, {"rotated": True}, b"old", self.old_secret
        )

        assert serializer.loads(value) is None
        assert inner.loads_count == 0

    @pytest.mark.parametrize(
        "value",
        [
            b"",
            b"\x01",
            pickle.dumps({"bare": True}),
        ],
        ids=["empty", "truncated", "bare-payload"],
    )
    def test_invalid_values_are_rejected_without_fallback(self, value):
        inner = CountingSerializer()
        serializer = self.make(inner)

        assert serializer.loads(value) is None
        assert inner.loads_count == 0

    def test_unknown_version_is_rejected(self):
        inner = CountingSerializer()
        serializer = self.make(inner)
        value = self.envelope(
            serializer, {"ok": True}, b"new", self.current_secret, version=2
        )

        assert serializer.loads(value) is None
        assert inner.loads_count == 0

    def test_empty_payload_is_rejected(self):
        inner = CountingSerializer()
        serializer = self.make(inner)
        key_id = b"new"
        header = struct.pack(">BBI", 1, len(key_id), 0)
        signed_data = header + key_id
        value = signed_data + self._signature_for_test(signed_data)

        assert serializer.loads(value) is None
        assert inner.loads_count == 0

    def _signature_for_test(self, signed_data):
        return hmac.new(self.current_secret, signed_data, sha256).digest()

    def test_truncated_stream_is_rejected(self):
        inner = CountingSerializer()
        serializer = self.make(inner)
        value = serializer.dumps({"ok": True})

        assert serializer.load(io.BytesIO(value[:-1])) is None
        assert inner.load_count == 0

    def test_unknown_key_id_is_rejected(self):
        inner = CountingSerializer()
        serializer = self.make(inner)
        value = self.envelope(
            serializer, {"ok": True}, b"bad", self.current_secret
        )

        assert serializer.loads(value) is None
        assert inner.loads_count == 0

    def test_payload_size_limit_is_enforced_before_loads(self):
        inner = CountingSerializer()
        serializer = self.make(inner, max_payload_size=4)

        assert serializer.dumps("12345") is None
        assert inner.loads_count == 0

    def test_from_keys_accepts_configured_old_key(self):
        inner = CountingSerializer()
        serializer = HMACSerializer.from_keys(
            inner,
            {b"new": self.current_secret, b"old": self.old_secret},
            b"new",
        )
        value = self.envelope(
            serializer, {"rotated": True}, b"old", self.old_secret
        )

        assert serializer.loads(value) == {"rotated": True}
        assert inner.loads_count == 1

    def test_from_keys_rejects_duplicate_normalized_key_id(self):
        with pytest.raises(ValueError, match="duplicate key id"):
            AuthenticatedSerializer.from_keys(
                CountingSerializer(),
                {b"new": self.current_secret, "new": b"other-secret"},
                b"new",
            )
