import io
import pickle
import struct

import pytest

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


class CountingDeserializer:
    def __init__(self):
        self.load_count = 0

    def dumps(self, value, protocol=pickle.HIGHEST_PROTOCOL):
        return pickle.dumps(value, protocol)

    def loads(self, value):
        self.load_count += 1
        return pickle.loads(value)


def signed_value(key_id, key, value="cached value"):
    serializer = HMACSerializer(
        BaseSerializer(), key, current_key_id=key_id
    )
    return bytearray(serializer.dumps(value))


def authenticated_load(serializer, value, use_stream=False):
    if use_stream:
        return serializer.load(io.BytesIO(bytes(value)))
    return serializer.loads(bytes(value))


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


class TestHMACSerializer:
    current_key = b"new-secret"
    old_key = b"old-secret"

    def make_serializer(self, inner=None, **kwargs):
        return HMACSerializer(
            inner,
            self.current_key,
            current_key_id=b"new",
            old_keys={b"old": self.old_key},
            **kwargs,
        )

    @pytest.mark.parametrize("use_stream", [False, True])
    def test_roundtrip(self, use_stream):
        inner = CountingDeserializer()
        serializer = self.make_serializer(inner)
        token = serializer.dumps({"answer": 42})

        assert authenticated_load(serializer, token, use_stream) == {"answer": 42}
        assert inner.load_count == 1

    @pytest.mark.parametrize("use_stream", [False, True])
    @pytest.mark.parametrize(
        ("mutation", "description"),
        [
            (lambda token: token.__setitem__(0, 2), "header"),
            (lambda token: token.__setitem__(-33, token[-33] ^ 1), "payload"),
            (lambda token: token.__setitem__(-1, token[-1] ^ 1), "signature"),
        ],
    )
    def test_tampered_values_do_not_reach_inner(
        self, mutation, description, use_stream
    ):
        inner = CountingDeserializer()
        serializer = self.make_serializer(inner)
        token = signed_value(b"old", self.old_key)
        mutation(token)

        assert authenticated_load(serializer, token, use_stream) is None
        assert inner.load_count == 0

    @pytest.mark.parametrize("use_stream", [False, True])
    def test_configured_old_key_restores_value(self, use_stream):
        inner = CountingDeserializer()
        serializer = self.make_serializer(inner)
        token = signed_value(b"old", self.old_key, {"rotated": True})

        assert authenticated_load(serializer, token, use_stream) == {"rotated": True}
        assert inner.load_count == 1

    @pytest.mark.parametrize("use_stream", [False, True])
    def test_unconfigured_old_key_is_rejected(self, use_stream):
        inner = CountingDeserializer()
        serializer = HMACSerializer(inner, self.current_key, current_key_id=b"new")
        token = signed_value(b"old", self.old_key)

        assert authenticated_load(serializer, token, use_stream) is None
        assert inner.load_count == 0

    @pytest.mark.parametrize("use_stream", [False, True])
    @pytest.mark.parametrize(
        "token_factory",
        [
            lambda: signed_value(b"old", b"old-secret")[:-1],
            lambda: signed_value(b"old", b"old-secret")[:10],
            lambda: bytearray(b"plain legacy value"),
            lambda: bytearray(pickle.dumps({"legacy": True})),
        ],
    )
    def test_truncated_and_bare_values_are_rejected(
        self, token_factory, use_stream
    ):
        inner = CountingDeserializer()
        serializer = self.make_serializer(inner)

        assert authenticated_load(serializer, token_factory(), use_stream) is None
        assert inner.load_count == 0

    @pytest.mark.parametrize("use_stream", [False, True])
    def test_unknown_version_is_rejected(self, use_stream):
        inner = CountingDeserializer()
        serializer = self.make_serializer(inner)
        token = signed_value(b"old", self.old_key)
        token[0] = 2

        assert authenticated_load(serializer, token, use_stream) is None
        assert inner.load_count == 0

    def test_new_values_use_current_key_id(self):
        serializer = self.make_serializer()
        token = serializer.dumps("value")

        assert token[10 : 10 + token[1]] == b"new"

    def test_write_rejects_oversized_payload(self):
        serializer = self.make_serializer(max_payload_size=4)

        with pytest.raises(ValueError):
            serializer.dumps("too large")

    @pytest.mark.parametrize("use_stream", [False, True])
    def test_read_rejects_declared_oversized_payload(self, use_stream):
        inner = CountingDeserializer()
        serializer = self.make_serializer(inner, max_payload_size=4)
        header = struct.pack("!BBQ", 1, 1, 5)
        token = header + b"x" + b"payload" + b"0" * 32

        assert authenticated_load(serializer, token, use_stream) is None
        assert inner.load_count == 0

    def test_rejection_logs_do_not_contain_keys(self, caplog):
        inner = CountingDeserializer()
        serializer = HMACSerializer(
            inner,
            b"super-secret-current",
            current_key_id=b"new",
            old_keys={b"old": b"super-secret-old"},
        )
        token = signed_value(b"old", b"super-secret-old")
        token[-1] ^= 1

        with caplog.at_level("WARNING"):
            assert serializer.loads(bytes(token)) is None

        assert "super-secret-current" not in caplog.text
        assert "super-secret-old" not in caplog.text
        assert inner.load_count == 0

    def test_file_system_cache_rejects_tampered_file(self, tmp_path):
        from cachelib import FileSystemCache

        inner = CountingDeserializer()
        cache = FileSystemCache(str(tmp_path), threshold=0)
        cache.serializer = HMACSerializer(inner, self.current_key)

        assert cache.set("answer", {"value": 42})
        assert cache.get("answer") == {"value": 42}
        assert inner.load_count == 1

        filename = cache._get_filename("answer")
        with open(filename, "r+b") as f:
            f.seek(-1, io.SEEK_END)
            signature_byte = f.read(1)[0]
            f.seek(-1, io.SEEK_END)
            f.write(bytes([signature_byte ^ 1]))

        assert cache.get("answer") is None
        assert inner.load_count == 1


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
