import random
import time
from typing import Any, Dict, List, Tuple

try:
    import torch
except Exception:  # Allows protocol unit tests in lightweight Python envs.
    torch = None

from .packet_protocol import (
    PEARLPacket,
    PacketFlags,
    PayloadType,
    deserialize_payload,
    fragment_payload,
    serialize_payload,
)


class ResearchNetworkSimulator:
    """
    Packet-level WAN simulator for cloud-edge experiments.

    It models propagation delay, bandwidth, jitter, MTU fragmentation, random or
    burst loss, packet reordering, ACK/retransmission cost, and protocol overhead.
    The public send() API mirrors NetworkSimulator so existing benchmarks can opt in
    without changing evaluation scripts.
    """

    def __init__(
        self,
        rtt_ms: float = 50.0,
        bandwidth_mbps: float = 100.0,
        packet_loss_rate: float = 0.0,
        jitter_ms: float = 5.0,
        mtu_bytes: int = 1500,
        reorder_rate: float = 0.0,
        timeout_ms: float = 200.0,
        max_retries: int = 3,
        loss_model: str = "random",
        burst_bad_prob: float = 0.35,
        burst_good_to_bad: float = 0.02,
        burst_bad_to_good: float = 0.30,
        jitter_model: str = "uniform",
        session_id: int = 1,
    ):
        self.rtt_ms = rtt_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.packet_loss_rate = packet_loss_rate
        self.jitter_ms = jitter_ms
        self.mtu_bytes = mtu_bytes
        self.reorder_rate = reorder_rate
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.loss_model = loss_model
        self.burst_bad_prob = burst_bad_prob
        self.burst_good_to_bad = burst_good_to_bad
        self.burst_bad_to_good = burst_bad_to_good
        self.jitter_model = jitter_model
        self.session_id = session_id

        self.seq = 1
        self.in_bad_state = False
        self.stats = {
            "packet_count": 0,
            "ack_count": 0,
            "packet_drop_count": 0,
            "retransmit_count": 0,
            "reordered_count": 0,
            "payload_bytes": 0,
            "header_bytes": 0,
            "wire_bytes": 0,
            "total_delay_injected_sec": 0.0,
            "timeout_count": 0,
        }

    def _payload_type_for(self, data: Any) -> PayloadType:
        if isinstance(data, dict):
            data_type = data.get("type")
            if data_type == "topk_quantized":
                return PayloadType.TOPK_LOGITS
            if data_type == "quantized":
                return PayloadType.QUANTIZED_LOGITS
            if data_type == "raw":
                return PayloadType.VERIFY_REQUEST
            if data_type == "verify_result":
                return PayloadType.VERIFY_RESULT
            if data_type == "fallback":
                return PayloadType.FALLBACK_SIGNAL
        if torch is not None and isinstance(data, torch.Tensor):
            return PayloadType.DRAFT_TOKENS
        return PayloadType.GENERIC

    def _sample_jitter(self) -> float:
        if self.jitter_ms <= 0:
            return 0.0
        if self.jitter_model == "normal":
            return random.gauss(0.0, self.jitter_ms / 2.0) / 1000.0
        if self.jitter_model == "lognormal":
            return (random.lognormvariate(0.0, 0.35) - 1.0) * self.jitter_ms / 1000.0
        return random.uniform(-self.jitter_ms, self.jitter_ms) / 1000.0

    def _packet_delay(self, wire_bytes: int) -> float:
        propagation = self.rtt_ms / 2000.0
        transmission = (wire_bytes * 8) / max(self.bandwidth_mbps * 1e6, 1.0)
        return max(0.0, propagation + transmission + self._sample_jitter())

    def _is_lost(self) -> bool:
        if self.loss_model == "gilbert_elliott":
            if self.in_bad_state:
                lost = random.random() < self.burst_bad_prob
                if random.random() < self.burst_bad_to_good:
                    self.in_bad_state = False
                return lost
            if random.random() < self.burst_good_to_bad:
                self.in_bad_state = True
                return random.random() < self.burst_bad_prob
            return random.random() < self.packet_loss_rate
        return random.random() < self.packet_loss_rate

    def _make_packets(self, payload: bytes, payload_type: PayloadType) -> List[PEARLPacket]:
        fragments = fragment_payload(payload, self.mtu_bytes)
        packets = []
        for i, fragment in enumerate(fragments):
            flags = 0
            if len(fragments) > 1:
                flags |= int(PacketFlags.FRAGMENT)
            if i == len(fragments) - 1:
                flags |= int(PacketFlags.LAST_FRAGMENT)
            packets.append(
                PEARLPacket.build(
                    session_id=self.session_id,
                    seq=self.seq,
                    flags=flags,
                    payload_type=payload_type,
                    payload=fragment,
                )
            )
            self.seq += 1
        return packets

    def _record_packet(self, packet: PEARLPacket, delay: float):
        packet_stats = packet.to_stats()
        self.stats["packet_count"] += 1
        self.stats["payload_bytes"] += packet_stats["payload_bytes"]
        self.stats["header_bytes"] += packet_stats["header_bytes"]
        self.stats["wire_bytes"] += packet_stats["wire_bytes"]
        self.stats["total_delay_injected_sec"] += delay

    def send(self, data: Any, simulate_delay: bool = True) -> Tuple[bool, Any]:
        payload = serialize_payload(data)
        payload_type = self._payload_type_for(data)
        packets = self._make_packets(payload, payload_type)

        if len(packets) > 1 and random.random() < self.reorder_rate:
            random.shuffle(packets)
            self.stats["reordered_count"] += 1

        received: Dict[int, bytes] = {}
        for packet in packets:
            attempts = 0
            delivered = False
            while attempts <= self.max_retries and not delivered:
                attempts += 1
                delay = self._packet_delay(packet.header_len + packet.payload_len)
                if self._is_lost():
                    self.stats["packet_drop_count"] += 1
                    if attempts <= self.max_retries:
                        self.stats["retransmit_count"] += 1
                        self.stats["timeout_count"] += 1
                        delay += self.timeout_ms / 1000.0
                    else:
                        if simulate_delay:
                            time.sleep(delay)
                        self.stats["total_delay_injected_sec"] += delay
                        return False, None
                else:
                    self._record_packet(packet, delay)
                    self.stats["ack_count"] += 1
                    if simulate_delay:
                        time.sleep(delay)
                    received[packet.seq] = packet.payload
                    delivered = True

        ordered_payload = b"".join(received[seq] for seq in sorted(received))
        return True, deserialize_payload(ordered_payload)

    def get_stats(self) -> Dict[str, float]:
        packet_count = max(self.stats["packet_count"], 1)
        wire_bytes = max(self.stats["wire_bytes"], 1)
        stats = dict(self.stats)
        stats.update(
            {
                "packet_loss_rate_actual": self.stats["packet_drop_count"] / packet_count,
                "protocol_overhead_ratio": self.stats["header_bytes"] / wire_bytes,
                "goodput_mbps": (
                    self.stats["payload_bytes"] * 8 / max(self.stats["total_delay_injected_sec"], 1e-9) / 1e6
                ),
                "mtu_bytes": self.mtu_bytes,
                "loss_model": self.loss_model,
                "jitter_model": self.jitter_model,
            }
        )
        return stats

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0
        self.in_bad_state = False
