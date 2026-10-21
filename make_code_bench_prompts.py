"""Write code-comprehension prompts for realistic speculative-decoding benches.

Technical exposition (make_bench_prompts.py) has unusually predictable
continuations, which flatters a drafter: measured DFlash2 acceptance was 97.3%
first-token and 6.31 mean acceptance length. Real assistant traffic is closer to
"read a body of code, then write the missing part", which has more high-entropy
surface for a drafter to miss.

Each prompt embeds a substantial, self-contained Python module with a removed
implementation, and asks for the missing piece. Sources are built inline (no
network, deterministic) and every prompt is distinct so prefix caching cannot
flatter the drafter.
"""

import argparse
import json

# Each entry: (module name, description of what is missing, module source with
# the target function/class body removed, the request text).
MODULES = [
    (
        "interval_map",
        "the `merge` method",
        '''
class IntervalMap:
    """Maps half-open intervals [start, end) to values, supporting lookup."""

    def __init__(self):
        self._starts = []
        self._ends = []
        self._values = []

    def insert(self, start, end, value):
        if end <= start:
            raise ValueError("empty interval")
        lo, hi = 0, len(self._starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._starts[mid] < start:
                lo = mid + 1
            else:
                hi = mid
        self._starts.insert(lo, start)
        self._ends.insert(lo, end)
        self._values.insert(lo, value)

    def lookup(self, point):
        lo, hi = 0, len(self._starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._starts[mid] <= point:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return None
        idx = lo - 1
        if point < self._ends[idx]:
            return self._values[idx]
        return None

    def merge(self):
        """Collapse adjacent/overlapping intervals that share a value.

        Intervals with the same value whose ranges touch or overlap must be
        joined into a single interval. Intervals keep sorted-by-start order.
        Returns the number of intervals removed.
        """
        # TODO: implement
        raise NotImplementedError
''',
        "merge() must join same-valued intervals that touch or overlap, keeping start order, and return how many were removed.",
    ),
    (
        "lru_ttl_cache",
        "the `get` method",
        '''
import time
from collections import OrderedDict


class LRUTTLCache:
    """LRU cache with per-entry TTL and a size bound."""

    def __init__(self, capacity, ttl_seconds):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._data = OrderedDict()

    def _now(self):
        return time.monotonic()

    def put(self, key, value):
        now = self._now()
        if key in self._data:
            del self._data[key]
        elif len(self._data) >= self.capacity:
            self._data.popitem(last=False)
        self._data[key] = (value, now + self.ttl)
        self._data.move_to_end(key)

    def _evict_expired(self, now):
        for key in [k for k, (_, exp) in self._data.items() if exp <= now]:
            del self._data[key]

    def get(self, key):
        """Return the cached value, or None when absent or expired.

        Expired entries must be treated as absent and must not be resurrected
        by an LRU touch. A hit refreshes recency.
        """
        # TODO: implement
        raise NotImplementedError

    def __len__(self):
        return len(self._data)
''',
        "get() must return None for absent or expired keys, evict expired entries, and refresh LRU recency only on a live hit.",
    ),
    (
        "token_bucket",
        "the `consume` method",
        '''
import time


class TokenBucket:
    """Token-bucket rate limiter with burst capacity."""

    def __init__(self, rate_per_sec, burst, clock=time.monotonic):
        if rate_per_sec <= 0:
            raise ValueError("rate must be positive")
        if burst <= 0:
            raise ValueError("burst must be positive")
        self.rate = float(rate_per_sec)
        self.burst = float(burst)
        self._tokens = float(burst)
        self._last = clock()
        self._clock = clock

    def _refill(self):
        now = self._clock()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last = now

    def consume(self, n=1):
        """Take n tokens if available.

        Returns True and deducts when the bucket holds enough; otherwise
        returns False and leaves the balance unchanged (no partial take).
        Must refill based on elapsed time first.
        """
        # TODO: implement
        raise NotImplementedError
''',
        "consume() must refill from elapsed time, take all n tokens atomically or none, and never let the balance exceed burst.",
    ),
    (
        "sparse_row",
        "the `set` method",
        '''
class SparseRow:
    """Sparse row vector stored as sorted (index, value) pairs."""

    def __init__(self, size):
        self.size = size
        self._idx = []
        self._val = []

    def _find(self, index):
        lo, hi = 0, len(self._idx)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._idx[mid] < index:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def get(self, index):
        pos = self._find(index)
        if pos < len(self._idx) and self._idx[pos] == index:
            return self._val[pos]
        return 0.0

    def set(self, index, value):
        """Store value at index, keeping indices sorted and unique.

        Writing exactly 0.0 must remove the stored entry rather than store a
        zero, so that nnz() only counts structural nonzeros. Out-of-range
        indices raise IndexError.
        """
        # TODO: implement
        raise NotImplementedError

    def nnz(self):
        return len(self._idx)

    def dot(self, dense):
        return sum(v * dense[i] for i, v in zip(self._idx, self._val))
''',
        "set() must keep indices sorted and unique, delete the entry when writing 0.0, and raise IndexError out of range.",
    ),
]

REQUEST_TEMPLATES = [
    "Here is a Python module. {desc}\n\nImplement it, explaining the edge cases you handle.",
    "Read this module and {desc}\n\nWrite the implementation with a short justification for each choice.",
    "The following module is missing one method. {desc}\n\nProvide the implementation and note any invariants it preserves.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = []
    for i in range(args.num_prompts):
        name, missing, source, desc = MODULES[i % len(MODULES)]
        template = REQUEST_TEMPLATES[(i // len(MODULES)) % len(REQUEST_TEMPLATES)]
        request = template.format(desc=desc)
        prompt = (
            f"{request}\n\n"
            f"```python\n{source.strip()}\n```\n\n"
            f"The missing piece is {missing}. Target roughly {args.output_len} tokens."
        )
        # Unique per-prompt preamble: identical prefixes would be served from the
        # prefix cache, so later requests would skip prefill and make the
        # drafter look better than it is.
        unique = f"Task {i + 1} of {args.num_prompts}"
        data.append(
            {
                "id": f"code-{i:04d}",
                "conversations": [
                    {"from": "human", "value": f"{unique}: {prompt}"},
                    {
                        "from": "gpt",
                        "value": (
                            "Here is the implementation, with the edge cases and "
                            "invariants called out inline."
                        ),
                    },
                ],
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"wrote {len(data)} code prompts to {args.out} (sharegpt format)")


if __name__ == "__main__":
    main()
