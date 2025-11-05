"""
Gun.js Storage Manager for distributed encrypted storage.

This module provides secure, distributed storage capabilities using Gun.js concepts
for peer-to-peer synchronization, encrypted personal experiences, and selective sharing.
"""

import json
import time
import hashlib
import base64
import os
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import sqlite3
import threading
import queue
import logging
import socket
import struct
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryCategory(Enum):
    """Categories of memory with different sharing policies."""
    PRIVATE = "private"  # Robot's secrets, never shared
    FAMILY = "family"    # Shared with trusted siblings
    PUBLIC = "public"    # Shared with all robots


class TrustLevel(Enum):
    """Trust levels for peer robots."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    SIBLING = 4  # Full trust


@dataclass
class Memory:
    """Represents a stored memory."""
    id: str
    category: MemoryCategory
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    encrypted: bool = False
    owner_id: str = ""
    tags: List[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    shared_with: Set[str] = field(default_factory=set)


@dataclass
class Peer:
    """Represents a peer robot."""
    id: str
    address: str
    port: int
    trust_level: TrustLevel
    last_seen: float = field(default_factory=time.time)
    shared_memories: int = 0
    public_key: Optional[str] = None


class GunStorageManager:
    """
    Distributed encrypted storage manager inspired by Gun.js.
    """
    
    def __init__(self, robot_id: str, config: Optional[Dict] = None):
        """
        Initialize the Gun storage manager.
        
        Args:
            robot_id: Unique identifier for this robot
            config: Configuration dictionary
        """
        self.robot_id = robot_id
        self.config = config or {}
        
        # Storage paths
        self.storage_dir = self.config.get('storage_dir', f'./gun_storage/{robot_id}')
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Encryption setup
        self.master_key = self._get_or_create_master_key()
        self.fernet = Fernet(self.master_key)
        
        # Local storage (SQLite for persistence)
        self.db_path = os.path.join(self.storage_dir, 'memories.db')
        self._init_database()
        
        # In-memory cache
        self.memory_cache: Dict[str, Memory] = {}
        self.peer_cache: Dict[str, Peer] = {}
        
        # Peer management
        self.trusted_peers: Dict[str, Peer] = {}
        self.pending_sync: queue.Queue = queue.Queue()
        
        # Network settings
        self.listen_port = self.config.get('listen_port', 8765)
        self.broadcast_interval = self.config.get('broadcast_interval', 30)
        
        # Privacy settings
        self.auto_share_family = self.config.get('auto_share_family', True)
        self.auto_share_public = self.config.get('auto_share_public', True)
        self.encryption_required = self.config.get('encryption_required', True)
        
        # Start background services
        self.running = False
        self.sync_thread = None
        self.listen_thread = None
        
        logger.info(f"GunStorageManager initialized for robot {robot_id}")
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key."""
        key_file = os.path.join(self.storage_dir, '.master.key')
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            
            # Save securely (in production, use proper key management)
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            return key
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp REAL NOT NULL,
                encrypted INTEGER NOT NULL,
                owner_id TEXT NOT NULL,
                tags TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                shared_with TEXT
            )
        ''')
        
        # Create peers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS peers (
                id TEXT PRIMARY KEY,
                address TEXT NOT NULL,
                port INTEGER NOT NULL,
                trust_level INTEGER NOT NULL,
                last_seen REAL NOT NULL,
                shared_memories INTEGER DEFAULT 0,
                public_key TEXT
            )
        ''')
        
        # Create sync log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                peer_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp REAL NOT NULL,
                success INTEGER NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_private_memory(self, data: Dict[str, Any], 
                            tags: Optional[List[str]] = None) -> str:
        """
        Store a private memory (never shared).
        
        Args:
            data: Memory data to store
            tags: Optional tags for categorization
            
        Returns:
            Memory ID
        """
        memory_id = self._generate_memory_id(data)
        
        # Encrypt private memories
        encrypted_data = self._encrypt_data(data)
        
        memory = Memory(
            id=memory_id,
            category=MemoryCategory.PRIVATE,
            data=encrypted_data,
            encrypted=True,
            owner_id=self.robot_id,
            tags=tags or []
        )
        
        # Store in database
        self._save_memory_to_db(memory)
        
        # Cache
        self.memory_cache[memory_id] = memory
        
        logger.info(f"Stored private memory: {memory_id}")
        return memory_id
    
    def store_shared_memory(self, data: Dict[str, Any],
                          category: MemoryCategory = MemoryCategory.FAMILY,
                          tags: Optional[List[str]] = None) -> str:
        """
        Store a memory that can be shared with others.
        
        Args:
            data: Memory data to store
            category: Sharing category
            tags: Optional tags
            
        Returns:
            Memory ID
        """
        memory_id = self._generate_memory_id(data)
        
        # Encrypt if needed
        if category == MemoryCategory.FAMILY and self.encryption_required:
            stored_data = self._encrypt_data(data)
            encrypted = True
        else:
            stored_data = data
            encrypted = False
        
        memory = Memory(
            id=memory_id,
            category=category,
            data=stored_data,
            encrypted=encrypted,
            owner_id=self.robot_id,
            tags=tags or []
        )
        
        # Store in database
        self._save_memory_to_db(memory)
        
        # Cache
        self.memory_cache[memory_id] = memory
        
        # Queue for synchronization
        if category in [MemoryCategory.FAMILY, MemoryCategory.PUBLIC]:
            self.pending_sync.put(('share', memory_id))
        
        logger.info(f"Stored {category.value} memory: {memory_id}")
        return memory_id
    
    def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Memory data or None if not found
        """
        # Check cache first
        if memory_id in self.memory_cache:
            memory = self.memory_cache[memory_id]
        else:
            # Load from database
            memory = self._load_memory_from_db(memory_id)
            if memory:
                self.memory_cache[memory_id] = memory
        
        if not memory:
            return None
        
        # Update access tracking
        memory.access_count += 1
        memory.last_accessed = time.time()
        
        # Decrypt if needed
        if memory.encrypted:
            try:
                decrypted_data = self._decrypt_data(memory.data)
                return decrypted_data
            except Exception as e:
                logger.error(f"Failed to decrypt memory {memory_id}: {e}")
                return None
        
        return memory.data
    
    def share_with_sibling(self, memory_id: str, sibling_id: str) -> bool:
        """
        Share a specific memory with a sibling robot.
        
        Args:
            memory_id: Memory to share
            sibling_id: Sibling robot ID
            
        Returns:
            True if shared successfully
        """
        if sibling_id not in self.trusted_peers:
            logger.warning(f"Cannot share with untrusted peer: {sibling_id}")
            return False
        
        peer = self.trusted_peers[sibling_id]
        if peer.trust_level.value < TrustLevel.HIGH.value:
            logger.warning(f"Insufficient trust level for peer: {sibling_id}")
            return False
        
        memory = self.memory_cache.get(memory_id) or self._load_memory_from_db(memory_id)
        if not memory:
            logger.error(f"Memory not found: {memory_id}")
            return False
        
        # Don't share private memories
        if memory.category == MemoryCategory.PRIVATE:
            logger.warning("Cannot share private memory")
            return False
        
        # Send memory to sibling
        success = self._send_memory_to_peer(memory, peer)
        
        if success:
            memory.shared_with.add(sibling_id)
            self._update_memory_in_db(memory)
            logger.info(f"Shared memory {memory_id} with {sibling_id}")
        
        return success
    
    def add_peer(self, peer_id: str, address: str, port: int,
                trust_level: TrustLevel = TrustLevel.LOW) -> bool:
        """
        Add a peer robot for synchronization.
        
        Args:
            peer_id: Peer robot ID
            address: Network address
            port: Network port
            trust_level: Initial trust level
            
        Returns:
            True if peer added successfully
        """
        peer = Peer(
            id=peer_id,
            address=address,
            port=port,
            trust_level=trust_level
        )
        
        # Save to database
        self._save_peer_to_db(peer)
        
        # Add to cache
        self.peer_cache[peer_id] = peer
        
        if trust_level.value >= TrustLevel.MEDIUM.value:
            self.trusted_peers[peer_id] = peer
        
        logger.info(f"Added peer {peer_id} with trust level {trust_level.name}")
        return True
    
    def update_trust_level(self, peer_id: str, new_level: TrustLevel):
        """
        Update trust level for a peer.
        
        Args:
            peer_id: Peer robot ID
            new_level: New trust level
        """
        if peer_id in self.peer_cache:
            peer = self.peer_cache[peer_id]
            peer.trust_level = new_level
            
            # Update trusted peers list
            if new_level.value >= TrustLevel.MEDIUM.value:
                self.trusted_peers[peer_id] = peer
            elif peer_id in self.trusted_peers:
                del self.trusted_peers[peer_id]
            
            # Update database
            self._update_peer_in_db(peer)
            
            logger.info(f"Updated trust level for {peer_id} to {new_level.name}")
    
    def start_sync(self):
        """Start background synchronization."""
        if self.running:
            return
        
        self.running = True
        
        # Start sync thread
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_worker, daemon=True)
        self.listen_thread.start()
        
        logger.info("Background synchronization started")
    
    def stop_sync(self):
        """Stop background synchronization."""
        self.running = False
        
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        
        if self.listen_thread:
            self.listen_thread.join(timeout=5)
        
        logger.info("Background synchronization stopped")
    
    def _sync_worker(self):
        """Background worker for synchronization."""
        while self.running:
            try:
                # Process pending sync queue
                if not self.pending_sync.empty():
                    action, memory_id = self.pending_sync.get(timeout=1)
                    
                    if action == 'share':
                        self._broadcast_memory(memory_id)
                
                # Periodic sync with peers
                self._sync_with_peers()
                
                # Sleep interval
                time.sleep(self.broadcast_interval)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Sync worker error: {e}")
    
    def _listen_worker(self):
        """Background worker for listening to peer updates."""
        try:
            # Create listening socket
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind(('0.0.0.0', self.listen_port))
            listen_socket.listen(5)
            listen_socket.settimeout(1.0)
            
            logger.info(f"Listening for peer connections on port {self.listen_port}")
            
            while self.running:
                try:
                    conn, addr = listen_socket.accept()
                    # Handle connection in separate thread
                    threading.Thread(
                        target=self._handle_peer_connection,
                        args=(conn, addr),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Listen error: {e}")
            
            listen_socket.close()
            
        except Exception as e:
            logger.error(f"Listen worker error: {e}")
    
    def _handle_peer_connection(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle incoming peer connection."""
        try:
            # Receive data
            data_size = struct.unpack('>I', conn.recv(4))[0]
            data = b''
            while len(data) < data_size:
                chunk = conn.recv(min(4096, data_size - len(data)))
                if not chunk:
                    break
                data += chunk
            
            # Deserialize
            message = pickle.loads(data)
            
            # Process message
            if message['type'] == 'memory_share':
                self._receive_shared_memory(message)
            elif message['type'] == 'sync_request':
                self._handle_sync_request(message, conn)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error handling peer connection: {e}")
            conn.close()
    
    def _broadcast_memory(self, memory_id: str):
        """Broadcast a memory to appropriate peers."""
        memory = self.memory_cache.get(memory_id) or self._load_memory_from_db(memory_id)
        if not memory:
            return
        
        # Determine target peers based on category
        target_peers = []
        
        if memory.category == MemoryCategory.PUBLIC:
            target_peers = list(self.peer_cache.values())
        elif memory.category == MemoryCategory.FAMILY:
            target_peers = [p for p in self.trusted_peers.values() 
                          if p.trust_level.value >= TrustLevel.HIGH.value]
        
        # Send to each peer
        for peer in target_peers:
            self._send_memory_to_peer(memory, peer)
    
    def _send_memory_to_peer(self, memory: Memory, peer: Peer) -> bool:
        """Send a memory to a specific peer."""
        try:
            # Prepare message
            message = {
                'type': 'memory_share',
                'sender_id': self.robot_id,
                'memory': {
                    'id': memory.id,
                    'category': memory.category.value,
                    'data': memory.data,
                    'timestamp': memory.timestamp,
                    'encrypted': memory.encrypted,
                    'owner_id': memory.owner_id,
                    'tags': memory.tags
                }
            }
            
            # Serialize
            data = pickle.dumps(message)
            
            # Send to peer
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((peer.address, peer.port))
            
            # Send size first, then data
            sock.send(struct.pack('>I', len(data)))
            sock.send(data)
            
            sock.close()
            
            # Log sync
            self._log_sync(peer.id, memory.id, 'share', True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send memory to peer {peer.id}: {e}")
            self._log_sync(peer.id, memory.id, 'share', False)
            return False
    
    def _receive_shared_memory(self, message: Dict[str, Any]):
        """Receive and store a shared memory from a peer."""
        sender_id = message['sender_id']
        memory_data = message['memory']
        
        # Check trust level
        if sender_id not in self.peer_cache:
            logger.warning(f"Received memory from unknown peer: {sender_id}")
            return
        
        peer = self.peer_cache[sender_id]
        
        # Apply trust-based filtering
        category = MemoryCategory(memory_data['category'])
        
        if category == MemoryCategory.PRIVATE:
            logger.warning(f"Peer {sender_id} tried to share private memory")
            return
        
        if category == MemoryCategory.FAMILY and peer.trust_level.value < TrustLevel.HIGH.value:
            logger.warning(f"Insufficient trust for family memory from {sender_id}")
            return
        
        # Create memory object
        memory = Memory(
            id=memory_data['id'],
            category=category,
            data=memory_data['data'],
            timestamp=memory_data['timestamp'],
            encrypted=memory_data['encrypted'],
            owner_id=memory_data['owner_id'],
            tags=memory_data['tags']
        )
        
        # Store if not already present
        if memory.id not in self.memory_cache:
            self._save_memory_to_db(memory)
            self.memory_cache[memory.id] = memory
            logger.info(f"Received and stored memory {memory.id} from {sender_id}")
        
        # Update peer stats
        peer.shared_memories += 1
        peer.last_seen = time.time()
        self._update_peer_in_db(peer)
    
    def _sync_with_peers(self):
        """Synchronize with all trusted peers."""
        for peer in self.trusted_peers.values():
            try:
                # Request sync
                message = {
                    'type': 'sync_request',
                    'sender_id': self.robot_id,
                    'last_sync': peer.last_seen
                }
                
                # Send sync request
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect((peer.address, peer.port))
                
                data = pickle.dumps(message)
                sock.send(struct.pack('>I', len(data)))
                sock.send(data)
                
                # Receive response
                response_size = struct.unpack('>I', sock.recv(4))[0]
                response_data = b''
                while len(response_data) < response_size:
                    chunk = sock.recv(min(4096, response_size - len(response_data)))
                    if not chunk:
                        break
                    response_data += chunk
                
                response = pickle.loads(response_data)
                
                # Process new memories
                for memory_data in response.get('memories', []):
                    self._receive_shared_memory({
                        'sender_id': peer.id,
                        'memory': memory_data
                    })
                
                sock.close()
                
                # Update peer last seen
                peer.last_seen = time.time()
                self._update_peer_in_db(peer)
                
            except Exception as e:
                logger.error(f"Sync with peer {peer.id} failed: {e}")
    
    def _handle_sync_request(self, message: Dict[str, Any], conn: socket.socket):
        """Handle sync request from a peer."""
        sender_id = message['sender_id']
        last_sync = message.get('last_sync', 0)
        
        if sender_id not in self.peer_cache:
            conn.close()
            return
        
        peer = self.peer_cache[sender_id]
        
        # Get memories to share based on trust level
        memories_to_share = []
        
        for memory in self.memory_cache.values():
            # Check if memory should be shared
            if memory.timestamp <= last_sync:
                continue
            
            if memory.category == MemoryCategory.PRIVATE:
                continue
            
            if memory.category == MemoryCategory.PUBLIC:
                memories_to_share.append(memory)
            elif (memory.category == MemoryCategory.FAMILY and 
                  peer.trust_level.value >= TrustLevel.HIGH.value):
                memories_to_share.append(memory)
        
        # Prepare response
        response = {
            'memories': [
                {
                    'id': m.id,
                    'category': m.category.value,
                    'data': m.data,
                    'timestamp': m.timestamp,
                    'encrypted': m.encrypted,
                    'owner_id': m.owner_id,
                    'tags': m.tags
                } for m in memories_to_share
            ]
        }
        
        # Send response
        data = pickle.dumps(response)
        conn.send(struct.pack('>I', len(data)))
        conn.send(data)
    
    def query_memories(self, tags: Optional[List[str]] = None,
                      category: Optional[MemoryCategory] = None,
                      owner: Optional[str] = None) -> List[Memory]:
        """
        Query memories based on criteria.
        
        Args:
            tags: Filter by tags
            category: Filter by category
            owner: Filter by owner
            
        Returns:
            List of matching memories
        """
        results = []
        
        # Search all memories
        all_memories = list(self.memory_cache.values())
        
        # Load from DB if cache is incomplete
        if len(all_memories) < 100:  # Arbitrary threshold
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM memories')
            for row in cursor.fetchall():
                memory_id = row[0]
                if memory_id not in self.memory_cache:
                    memory = self._load_memory_from_db(memory_id)
                    if memory:
                        all_memories.append(memory)
            conn.close()
        
        # Apply filters
        for memory in all_memories:
            if category and memory.category != category:
                continue
            
            if owner and memory.owner_id != owner:
                continue
            
            if tags:
                if not any(tag in memory.tags for tag in tags):
                    continue
            
            results.append(memory)
        
        return results
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count memories by category
        stats = {}
        for category in MemoryCategory:
            cursor.execute('SELECT COUNT(*) FROM memories WHERE category = ?', 
                         (category.value,))
            stats[f'{category.value}_memories'] = cursor.fetchone()[0]
        
        # Count peers
        cursor.execute('SELECT COUNT(*) FROM peers')
        stats['total_peers'] = cursor.fetchone()[0]
        
        # Count trusted peers
        stats['trusted_peers'] = len(self.trusted_peers)
        
        # Get sync stats
        cursor.execute('SELECT COUNT(*) FROM sync_log WHERE success = 1')
        stats['successful_syncs'] = cursor.fetchone()[0]
        
        conn.close()
        
        # Add cache stats
        stats['cached_memories'] = len(self.memory_cache)
        stats['pending_sync'] = self.pending_sync.qsize()
        stats['is_syncing'] = self.running
        
        return stats
    
    # Helper methods
    def _generate_memory_id(self, data: Dict[str, Any]) -> str:
        """Generate unique memory ID."""
        content = json.dumps(data, sort_keys=True)
        timestamp = str(time.time())
        combined = f"{self.robot_id}_{content}_{timestamp}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt data dictionary."""
        json_data = json.dumps(data)
        encrypted = self.fernet.encrypt(json_data.encode())
        return {'encrypted': base64.b64encode(encrypted).decode()}
    
    def _decrypt_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt data dictionary."""
        if 'encrypted' not in encrypted_data:
            return encrypted_data
        
        encrypted = base64.b64decode(encrypted_data['encrypted'])
        decrypted = self.fernet.decrypt(encrypted)
        return json.loads(decrypted.decode())
    
    def _save_memory_to_db(self, memory: Memory):
        """Save memory to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO memories 
            (id, category, data, timestamp, encrypted, owner_id, tags, 
             access_count, last_accessed, shared_with)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.id,
            memory.category.value,
            json.dumps(memory.data),
            memory.timestamp,
            int(memory.encrypted),
            memory.owner_id,
            json.dumps(memory.tags),
            memory.access_count,
            memory.last_accessed,
            json.dumps(list(memory.shared_with))
        ))
        
        conn.commit()
        conn.close()
    
    def _load_memory_from_db(self, memory_id: str) -> Optional[Memory]:
        """Load memory from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return Memory(
            id=row[0],
            category=MemoryCategory(row[1]),
            data=json.loads(row[2]),
            timestamp=row[3],
            encrypted=bool(row[4]),
            owner_id=row[5],
            tags=json.loads(row[6]) if row[6] else [],
            access_count=row[7],
            last_accessed=row[8],
            shared_with=set(json.loads(row[9])) if row[9] else set()
        )
    
    def _update_memory_in_db(self, memory: Memory):
        """Update memory in database."""
        self._save_memory_to_db(memory)
    
    def _save_peer_to_db(self, peer: Peer):
        """Save peer to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO peers 
            (id, address, port, trust_level, last_seen, shared_memories, public_key)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            peer.id,
            peer.address,
            peer.port,
            peer.trust_level.value,
            peer.last_seen,
            peer.shared_memories,
            peer.public_key
        ))
        
        conn.commit()
        conn.close()
    
    def _update_peer_in_db(self, peer: Peer):
        """Update peer in database."""
        self._save_peer_to_db(peer)
    
    def _log_sync(self, peer_id: str, memory_id: str, action: str, success: bool):
        """Log synchronization event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sync_log (peer_id, memory_id, action, timestamp, success)
            VALUES (?, ?, ?, ?, ?)
        ''', (peer_id, memory_id, action, time.time(), int(success)))
        
        conn.commit()
        conn.close()
    
    def close(self):
        """Clean up resources."""
        self.stop_sync()
        
        # Save cache to database
        for memory in self.memory_cache.values():
            self._save_memory_to_db(memory)
        
        logger.info("GunStorageManager closed")