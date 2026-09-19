"""
Spray and Wait — DTN fallback routing for extreme fragmentation.

Used when Q-routing fails because no connected path exists.
Provides eventual delivery via store-carry-forward.

Reference:
    Spyropoulos et al., "Spray and Wait: An Efficient Routing Scheme
    for Intermittently Connected Mobile Networks", WDTN 2005
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class SprayPacket:
    """Packet in spray-and-wait mode with copy budget."""
    packet_id: int
    source_id: int
    destination_id: int
    creation_time: float
    copies: int = 6       # L parameter
    ttl: float = 300.0    # seconds
    holders: List[int] = field(default_factory=list)


class SprayAndWait:
    """
    Spray and Wait routing protocol.

    Spray phase: Source distributes L/2 copies to first L/2 distinct relays.
    Wait phase: Each copy holder waits for direct contact with destination.
    """

    def __init__(self, spray_copies: int = 6, packet_ttl: float = 300.0,
                 connectivity_threshold: float = 0.3):
        self.spray_copies = spray_copies
        self.packet_ttl = packet_ttl
        self.connectivity_threshold = connectivity_threshold
        self.spray_buffer: Dict[int, SprayPacket] = {}  # packet_id → SprayPacket
        self.delivered: List[int] = []

    def should_activate(self, connectivity_ratio: float) -> bool:
        """Check if connectivity is low enough to activate DTN fallback."""
        return connectivity_ratio < self.connectivity_threshold

    def create_spray_packet(self, packet_id: int, source_id: int,
                            destination_id: int, current_time: float) -> SprayPacket:
        """Create a new spray packet with initial copy budget."""
        pkt = SprayPacket(
            packet_id=packet_id,
            source_id=source_id,
            destination_id=destination_id,
            creation_time=current_time,
            copies=self.spray_copies,
            ttl=self.packet_ttl,
            holders=[source_id],
        )
        self.spray_buffer[packet_id] = pkt
        return pkt

    def on_encounter(self, packet_id: int, holder_id: int,
                     encountered_id: int, current_time: float) -> Optional[str]:
        """
        Handle encounter event for spray-and-wait packet.

        Returns:
            'delivered' if packet reached destination,
            'sprayed' if copies were shared,
            'waiting' if in wait phase,
            None if packet expired or not found.
        """
        if packet_id not in self.spray_buffer:
            return None

        pkt = self.spray_buffer[packet_id]

        # Check TTL
        if current_time - pkt.creation_time > pkt.ttl:
            del self.spray_buffer[packet_id]
            return None

        # Direct delivery check
        if encountered_id == pkt.destination_id:
            self.delivered.append(packet_id)
            del self.spray_buffer[packet_id]
            return 'delivered'

        # Spray phase: share copies
        if pkt.copies > 1 and encountered_id not in pkt.holders:
            copies_to_give = pkt.copies // 2
            pkt.copies -= copies_to_give
            pkt.holders.append(encountered_id)
            return 'sprayed'

        # Wait phase: only direct delivery
        return 'waiting'

    def expire_old_packets(self, current_time: float):
        """Remove expired packets from buffer."""
        expired = [
            pid for pid, pkt in self.spray_buffer.items()
            if current_time - pkt.creation_time > pkt.ttl
        ]
        for pid in expired:
            del self.spray_buffer[pid]

    def get_buffer_size(self) -> int:
        """Number of packets in spray buffer."""
        return len(self.spray_buffer)
