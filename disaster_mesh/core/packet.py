"""
Packet/Message class with TTL, priority.
"""
import uuid

class Packet:
    def __init__(self, src_id: int, dest_id: int, priority: int = 1, ttl: float = 10.0, size_bytes: int = 256):
        self.packet_id = str(uuid.uuid4())
        self.src_id = src_id
        self.dest_id = dest_id
        self.priority = priority # 3: SOS, 2: Location, 1: Text
        self.ttl = ttl
        self.size_bytes = size_bytes
        self.creation_time = 0.0
        self.path = [src_id]
        self.delivered = False
        self.dropped = False
        self.delivery_time = -1.0
