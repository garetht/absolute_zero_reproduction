from base_buff import BaseBuffer, IOPair


class InductionBuffer(BaseBuffer):
    def generate_io_pairs(self, num_io_pairs: int, snippet: str) -> list[IOPair]: ...
