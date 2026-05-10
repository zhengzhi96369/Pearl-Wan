import hashlib
import pickle
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Iterable, List


class PayloadType(IntEnum):
    DRAFT_TOKENS = 1
    TOPK_LOGITS = 2
    QUANTIZED_LOGITS = 3
    VERIFY_REQUEST = 4
    VERIFY_RESULT = 5
    FALLBACK_SIGNAL = 6
    HEARTBEAT = 7
    ERROR = 8
    GENERIC = 255


class PacketFlags(IntEnum):
    ACK = 1
    RETRANSMIT = 2
    FRAGMENT = 4
    LAST_FRAGMENT = 8


@dataclass
class PEARLPacket:
    version: int
    session_id: int
    seq: int
    ack: int
    flags: int
    payload_type: int
    timestamp_us: int
    payload: bytes

    MAGIC = b"PWN1"
    HEADER_STRUCT = struct.Struct("!4sBQQQHHQI")

    @classmethod
    def build(
        cls,
        *,
        session_id: int,
        seq: int,
        ack: int = 0,
        flags: int = 0,
        payload_type: int = PayloadType.GENERIC,
        payload: bytes = b"",
    ) -> "PEARLPacket":
        return cls(
            version=1,
            session_id=session_id,
            seq=seq,
            ack=ack,
            flags=flags,
            payload_type=int(payload_type),
            timestamp_us=int(time.time() * 1_000_000),
            payload=payload,
        )

    @property
    def header_len(self) -> int:
        return self.HEADER_STRUCT.size + 32

    @property
    def payload_len(self) -> int:
        return len(self.payload)

    def encode(self) -> bytes:
        checksum = hashlib.sha256(self.payload).digest()
        header = self.HEADER_STRUCT.pack(
            self.MAGIC,
            self.version,
            self.session_id,
            self.seq,
            self.ack,
            self.flags,
            self.payload_type,
            self.timestamp_us,
            self.payload_len,
        )
        return header + checksum + self.payload

    @classmethod
    def decode(cls, wire: bytes) -> "PEARLPacket":
        min_len = cls.HEADER_STRUCT.size + 32
        if len(wire) < min_len:
            raise ValueError("wire packet is shorter than PEARL header")
        unpacked = cls.HEADER_STRUCT.unpack(wire[: cls.HEADER_STRUCT.size])
        magic, version, session_id, seq, ack, flags, payload_type, timestamp_us, payload_len = unpacked
        if magic != cls.MAGIC:
            raise ValueError("invalid PEARL packet magic")
        checksum = wire[cls.HEADER_STRUCT.size : min_len]
        payload = wire[min_len:]
        if len(payload) != payload_len:
            raise ValueError("payload length mismatch")
        if hashlib.sha256(payload).digest() != checksum:
            raise ValueError("payload checksum mismatch")
        return cls(version, session_id, seq, ack, flags, payload_type, timestamp_us, payload)

    def to_stats(self) -> Dict[str, int]:
        return {
            "header_bytes": self.header_len,
            "payload_bytes": self.payload_len,
            "wire_bytes": self.header_len + self.payload_len,
            "flags": self.flags,
            "payload_type": self.payload_type,
        }


def serialize_payload(data: Any) -> bytes:
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_payload(payload: bytes) -> Any:
    return pickle.loads(payload)


def fragment_payload(payload: bytes, mtu_bytes: int) -> List[bytes]:
    header_len = PEARLPacket.HEADER_STRUCT.size + 32
    max_payload = max(1, mtu_bytes - header_len)
    return [payload[i : i + max_payload] for i in range(0, len(payload), max_payload)] or [b""]


def iter_packet_wires(packets: Iterable[PEARLPacket]) -> Iterable[bytes]:
    for packet in packets:
        yield packet.encode()
