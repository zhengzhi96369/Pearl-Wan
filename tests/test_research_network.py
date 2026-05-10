import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.packet_protocol import PEARLPacket, PayloadType, deserialize_payload, serialize_payload
from src.research_network_simulator import ResearchNetworkSimulator


def test_packet_roundtrip():
    payload = serialize_payload({"hello": "world", "value": 3})
    packet = PEARLPacket.build(session_id=7, seq=1, payload_type=PayloadType.VERIFY_REQUEST, payload=payload)
    decoded = PEARLPacket.decode(packet.encode())
    assert decoded.session_id == 7
    assert decoded.seq == 1
    assert decoded.payload_type == PayloadType.VERIFY_REQUEST
    assert deserialize_payload(decoded.payload)["value"] == 3


def test_network_fragmentation_and_stats():
    net = ResearchNetworkSimulator(
        rtt_ms=0,
        bandwidth_mbps=1000,
        packet_loss_rate=0.0,
        jitter_ms=0,
        mtu_bytes=128,
        reorder_rate=1.0,
    )
    data = {"type": "verify_result", "payload": "x" * 4096}
    ok, received = net.send(data, simulate_delay=False)
    stats = net.get_stats()
    assert ok
    assert received == data
    assert stats["packet_count"] > 1
    assert stats["header_bytes"] > 0
    assert stats["protocol_overhead_ratio"] > 0


if __name__ == "__main__":
    test_packet_roundtrip()
    test_network_fragmentation_and_stats()
    print("research network tests passed")
