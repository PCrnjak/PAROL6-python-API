"""Timing metadata survives snapshots and malformed packets cannot invent it."""

import msgspec

from parol6.protocol.wire import StatusBuffer, decode_status_bin_into
from parol6.server.status_cache import StatusCache


def test_status_metadata_rejects_malformed_fields_and_clears_unavailable_metadata():
    cache = StatusCache()
    try:
        raw = cache.to_binary(session_id=2**64 - 1, seq=7, mono_time_ns=123456789)
        buffer = StatusBuffer()
        assert decode_status_bin_into(raw, buffer)
        frozen = buffer.copy()
        raw = cache.to_binary(session_id=9, seq=0, mono_time_ns=1)
        assert decode_status_bin_into(raw, buffer)
        assert (frozen.session_id, frozen.seq, frozen.mono_time_ns) == (
            2**64 - 1,
            7,
            123456789,
        )
        assert (buffer.session_id, buffer.seq, buffer.mono_time_ns) == (9, 0, 1)
        packet = msgspec.msgpack.decode(raw)
        # One slot later than the fields above them: the protocol version
        # leads the message.
        for field in (31, 32, 33):
            for invalid in (True, -1, 1.5, float("nan"), "1", None):
                changed = list(packet)
                changed[field] = invalid
                assert not decode_status_bin_into(
                    msgspec.msgpack.encode(changed), buffer
                )
        for length in (32, 33):
            assert not decode_status_bin_into(
                msgspec.msgpack.encode(packet[:length]), buffer
            )
        assert decode_status_bin_into(msgspec.msgpack.encode(packet[:31]), buffer)
        assert (buffer.session_id, buffer.seq, buffer.mono_time_ns) == (0, 0, 0)
    finally:
        cache.close()
