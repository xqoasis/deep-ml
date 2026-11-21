import numpy as np
import heapq

class VirtualKVCacheManager:
    def __init__(self, num_physical_blocks: int, block_size: int, hidden_dim: int):
        """
        Initialize the virtual memory KV cache manager.

        Args:
            num_physical_blocks: Total number of physical memory blocks
            block_size: Number of token slots per block
            hidden_dim: Dimension of each key/value vector
        """
        self.block_size = block_size
        # physical memory
        # [块数, 每块 token 数, hidden]，K 和 V 各一份
        self.k_memory = np.zeros((num_physical_blocks, block_size, hidden_dim))
        self.v_memory = np.zeros((num_physical_blocks, block_size, hidden_dim))

        # pblock_list[pbid[tid]] = {token_key, token_value}
        self.free_blocks = list(range(num_physical_blocks))
        self.seq_to_blocks = {}


    def allocate_block(self, seq_id: int) -> int:
        """Allocate a new physical block for seq_id. Return logical index or -1."""
        if not self.free_blocks:
            return -1

        # from free list pop one. allocate a new block and return the block id
        bid = heapq.heappop(self.free_blocks)
        seq_blocks = self.seq_to_blocks.setdefault(seq_id, [])
        seq_blocks.append(bid)
        return len(seq_blocks) - 1
        

    def write(self, seq_id: int, position: int, key: np.ndarray, value: np.ndarray) -> bool:
        """Write KV pair at token position. Auto-allocates blocks. Return success."""
        # logical
        logical_block_num, off = divmod(position, self.block_size)
        need = logical_block_num + 1 - len(self.seq_to_blocks.get(seq_id, []))

        if len(self.free_blocks) < need:
            return False

        # assign or fill
        for _ in range(need):
            self.allocate_block(seq_id)
        
        # get the new block id and position
        # write to that block
        block_id = self.seq_to_blocks[seq_id][logical_block_num]
        self.k_memory[block_id, off] = key
        self.v_memory[block_id, off] = value

        return True



    def read(self, seq_id: int, position: int):
        """Read KV pair at token position. Return (key, value) or (None, None)."""
        
        # get the block position
        logical_block_num, off = divmod(position, self.block_size)

        seq_blocks_ids = self.seq_to_blocks.get(seq_id)
        if seq_blocks_ids is None or logical_block_num >= len(seq_blocks_ids):
            return (None, None)
        blocks_id = seq_blocks_ids[logical_block_num]
        return self.k_memory[blocks_id, off], self.v_memory[blocks_id, off]
        

    def free_sequence(self, seq_id: int) -> int:
        """Free all blocks for seq_id. Return number of blocks freed."""
        removed_seq_blocks_ids = self.seq_to_blocks.pop(seq_id, [])
        # remove all this key value in pares
        # put all into free list
        for id in removed_seq_blocks_ids:
            self.k_memory[id] = 0
            self.v_memory[id] = 0
            heapq.heappush(self.free_blocks, id)
        return len(removed_seq_blocks_ids)


    def get_num_free_blocks(self) -> int:
        """Return number of unallocated physical blocks."""
        # return the stack
        return len(self.free_blocks)

    def get_page_table(self, seq_id: int) -> list:
        """Return list of physical block indices in logical order for seq_id."""
        return list(self.seq_to_blocks.get(seq_id, []))