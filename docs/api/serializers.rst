Serializers
===========

.. automodule:: cachelib.serializers
   :members:
   :undoc-members:
   :show-inheritance:

Authenticated file cache values
-------------------------------

Wrap a serializer with :class:`~cachelib.serializers.AuthenticatedSerializer`
when a cache directory may be writable by another local process. New values
are signed with the current HMAC key, while previous key ids can be passed
explicitly for verification during key rotation. Unauthenticated values,
unknown versions, unknown key ids, truncated values, and invalid signatures
are rejected without calling the wrapped deserializer.

.. code-block:: python

    from cachelib import FileSystemCache
    from cachelib.serializers import (
        AuthenticatedSerializer,
        FileSystemSerializer,
    )

    cache = FileSystemCache("/tmp/cache")
    cache.serializer = AuthenticatedSerializer(
        FileSystemSerializer(),
        b"current-secret",
        key_id=b"2026-10",
        old_keys={b"2026-09": b"previous-secret"},
    )
