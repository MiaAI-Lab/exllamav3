from __future__ import annotations
import heapq
import threading
import torch
from collections import deque
from ..constants import PAGE_SIZE


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class CPUPageCache:
    """
    Second-tier page cache in pinned system memory.

    Stores complete, hashed K/V pages evicted from the GPU page cache, keyed by the same chained page hash the
    PageTable uses. Each entry occupies one fixed-size slot holding the concatenated per-layer cache state of one
    page across all attached caches (main and draft cache, so restored pages remain valid for speculative
    decoding). Pages are pushed when PageTable.evict repurposes them and restored during allocation whenever a
    missing page hash is found here, replacing a prefill pass over the page with one host-to-device copy per
    layer tensor.

    All transfers are enqueued on each cache tensor's current stream. Pushes are therefore ordered before
    anything that could overwrite the dying page, restores are ordered before any kernel that could read the
    restored page, and slot recycling is race-free because a given byte range of a slot only ever moves over the
    stream of its fixed segment device.

    Eviction mirrors the GPU tier's policy: orphaned chains first, then whole trees, least recently used root
    first, pruned tail-first. A chain whose parent page is still live in the GPU page table counts as rooted.
    The order is consumed as a snapshot and rebuilt after at most max(64, max_slots/8) evictions, so recency
    updates between rebuilds are approximated.
    """

    def __init__(
        self,
        caches: list,
        max_size: int,
    ):
        """
        :param caches:
            List of Cache objects whose paged layers make up one page image, i.e. [cache] or
            [cache, draft_cache]. The model must be loaded (cache tensors allocated)

        :param max_size:
            Capacity in bytes of pinned system memory. Slots are allocated lazily as pages are pushed, so this
            is a ceiling, not an up-front allocation
        """

        # Segment table: one entry per paged cache tensor. Every cache tensor is page-major, so one page is the
        # contiguous slice tensor[page_index]
        self.segments = []
        offset = 0
        for cache in caches:
            for layer in cache.layers.values():
                for t in layer.get_tensors():
                    if t is None:
                        continue
                    assert t.device.type == "cuda", \
                        "Cannot build CPU page cache tier before the model (and its cache tensors) are loaded."
                    page_shape = tuple(t.shape[1:])
                    nbytes = t[0].numel() * t.element_size()
                    self.segments.append((t, offset, page_shape, t.dtype))
                    offset = _align(offset + nbytes, 256)
        assert self.segments, "No paged cache layers to attach CPU page cache tier to."

        self.slot_size = _align(offset, 4096)
        self.max_slots = int(max_size) // self.slot_size
        assert self.max_slots >= 2, \
            f"CPU page cache of {max_size} bytes is smaller than two pages ({self.slot_size} bytes per page)."

        self.pagetable = None

        # phash -> {slot, prev_hash, access_serial, tokens}
        self.entries = {}
        self.slot_slabs = []
        self.slot_views = []
        self.free_slots = deque()

        # Eviction order snapshot
        self._order = deque()
        self._order_pops = 0
        self._order_rebuild = max(64, self.max_slots // 8)

        self.metrics = {
            "pushes": 0,        # pages copied to the tier on GPU eviction
            "dedup_hits": 0,    # pushes skipped because the page was already stored
            "restores": 0,      # pages copied back into the GPU cache at allocation
            "evictions": 0,     # tier entries dropped to make room
            "cold_allocs": 0,   # pushes that had to pin a slab synchronously (spare pool was empty)
        }

        # Transfers run at PCIe speed, but pinning host memory only manages ~2.5 GB/s and serializes with copy
        # submission on the driver, so the full configured capacity is pinned up front by a background thread
        # (mirroring the GPU cache, whose full allocation is also committed at load). Pushes that outrun the
        # pinning thread early in the process fall back to pinning synchronously.
        self._spare = deque()
        self._spare_cond = threading.Condition()
        self._alloc_thread = threading.Thread(target = self._alloc_worker, daemon = True)
        self._alloc_thread.start()


    def attach(self, pagetable):
        self.pagetable = pagetable


    def __contains__(self, phash: bytes):
        return phash in self.entries


    def __len__(self):
        return len(self.entries)


    def _make_slab(self):
        slab = torch.empty((self.slot_size,), dtype = torch.uint8, pin_memory = True)
        views = []
        for t, offset, page_shape, dtype in self.segments:
            nbytes = t[0].numel() * t.element_size()
            views.append(slab[offset : offset + nbytes].view(dtype).view(page_shape))
        return slab, views


    def _alloc_worker(self):
        with torch.inference_mode():
            while True:
                with self._spare_cond:
                    while len(self.slot_slabs) + len(self._spare) >= self.max_slots:
                        self._spare_cond.wait()
                sv = self._make_slab()  # slow part, outside the lock
                with self._spare_cond:
                    self._spare.append(sv)


    def _new_slot(self, protect: set | None):
        if self.free_slots:
            return self.free_slots.popleft()
        if len(self.slot_slabs) < self.max_slots:
            with self._spare_cond:
                if self._spare:
                    sv = self._spare.popleft()
                    self.slot_slabs.append(sv[0])
                    self.slot_views.append(sv[1])
                    self._spare_cond.notify()
                    return len(self.slot_slabs) - 1
            sv = self._make_slab()
            self.metrics["cold_allocs"] += 1
            with self._spare_cond:
                self.slot_slabs.append(sv[0])
                self.slot_views.append(sv[1])
                self._spare_cond.notify()
                return len(self.slot_slabs) - 1
        return self._evict_one(protect)


    def _evict_one(self, protect: set | None):
        assert self.entries, "CPU page cache has no entries to evict (logic error)"
        deferred = 0
        while True:
            if not self._order or self._order_pops >= self._order_rebuild:
                self._build_order()
                deferred = 0
            h = self._order.popleft()
            self._order_pops += 1
            # Entries claimed by the allocation in progress are skipped unless everything is protected
            if protect and h in protect and deferred < len(self.entries):
                self._order.append(h)
                deferred += 1
                continue
            e = self.entries.pop(h, None)
            if e is not None:
                self.metrics["evictions"] += 1
                return e["slot"]


    def _build_order(self):
        """
        Snapshot of the eviction order over current entries; see class docstring.
        """
        entries = self.entries
        children = {h: [] for h in entries}
        roots = []
        orphan_roots = []
        pt = self.pagetable
        for h, e in entries.items():
            ph = e["prev_hash"]
            if ph is not None and ph in children:
                children[ph].append(h)
            elif ph is None or (pt is not None and pt.get_live_page(ph) is not None):
                roots.append(h)
            else:
                orphan_roots.append(h)

        order = []
        def prune(root):
            remaining = {}
            heap = []
            stack = [root]
            while stack:
                h = stack.pop()
                n = len(children[h])
                remaining[h] = n
                if n == 0:
                    e = entries[h]
                    heapq.heappush(heap, (e["access_serial"], e["slot"], h))
                stack.extend(children[h])
            while heap:
                _, _, h = heapq.heappop(heap)
                order.append(h)
                ph = entries[h]["prev_hash"]
                if ph is not None and ph in remaining:
                    remaining[ph] -= 1
                    if remaining[ph] == 0:
                        e = entries[ph]
                        heapq.heappush(heap, (e["access_serial"], e["slot"], ph))

        serial = lambda h: entries[h]["access_serial"]
        for root in sorted(orphan_roots, key = serial):
            prune(root)
        for root in sorted(roots, key = serial):
            prune(root)

        self._order = deque(order)
        self._order_pops = 0


    def store(self, page, serial: int, protect: set | None = None):
        """
        Copy a dying page's cache state into the tier (device-to-host, async on the current stream). Duplicate
        hashes only refresh the entry's recency.
        """
        e = self.entries.get(page.phash)
        if e is not None:
            e["access_serial"] = serial
            self.metrics["dedup_hits"] += 1
            return
        slot = self._new_slot(protect)
        for v, (t, _, _, _) in zip(self.slot_views[slot], self.segments):
            v.copy_(t[page.page_index], non_blocking = True)
        self.entries[page.phash] = {
            "slot": slot,
            "prev_hash": page.prev_hash,
            "access_serial": serial,
            "tokens": page.sequence.clone(),
        }
        self.metrics["pushes"] += 1


    def fetch(self, phash: bytes, page_index: int, serial: int) -> dict:
        """
        Copy a stored page back into the GPU cache at page_index (host-to-device, async on the current stream).
        The entry remains in the tier; the restored copy may well be evicted again before this one goes stale.
        Returns the entry so the caller can restore page metadata (token IDs).
        """
        e = self.entries[phash]
        e["access_serial"] = serial
        for v, (t, _, _, _) in zip(self.slot_views[e["slot"]], self.segments):
            t[page_index].copy_(v, non_blocking = True)
        self.metrics["restores"] += 1
        return e


# ---------------------------------------------------------------------------------------------------------------
# Tensor-parallel support (local patch 0003)
#
# Under TP the target cache tensors live in the TP workers (one shard per device; the output device's shard lives
# in this process behind a pseudo-worker), so they can't be copied from here. TPCPUPageCache keeps the whole
# index (hashes, recency, eviction) in the master exactly as CPUPageCache does, but each worker holds the pinned
# host slabs for its own shard, keyed by the master's slot number. Stores and fetches are dispatched to every
# worker through the same queue as forward passes, so they stay ordered with respect to the kernels that
# overwrite or read the page. Non-TP caches (e.g. a DFlash2 draft cache) are handled locally by the base class.

def mp_cpu_tier_page_bytes(local_context: dict, cache_id: int):
    total = 0
    for module in local_context["kv_modules"]:
        layer = module.tp_cache_lookup.get(cache_id)
        if layer is None:
            continue
        for t in layer.get_tensors():
            if t is not None:
                total += _align(t[0].numel() * t.element_size(), 256)
    return total


def _mp_cpu_tier_tensors(local_context: dict, cache_id: int):
    key = ("cpu_tier_tensors", cache_id)
    ts = local_context.get(key)
    if ts is None:
        ts = []
        for module in local_context["kv_modules"]:
            layer = module.tp_cache_lookup.get(cache_id)
            if layer is None:
                continue
            ts += [t for t in layer.get_tensors() if t is not None]
        local_context[key] = ts
    return ts


def _mp_cpu_tier_views(local_context: dict, cache_id: int, slot: int):
    slabs = local_context.setdefault(("cpu_tier_slabs", cache_id), {})
    views = slabs.get(slot)
    if views is None:
        ts = _mp_cpu_tier_tensors(local_context, cache_id)
        sizes = [_align(t[0].numel() * t.element_size(), 256) for t in ts]
        slab = torch.empty((max(sum(sizes), 1),), dtype = torch.uint8, pin_memory = True)
        views, offset = [], 0
        for t, n in zip(ts, sizes):
            nb = t[0].numel() * t.element_size()
            views.append(slab[offset : offset + nb].view(t.dtype).view(tuple(t.shape[1:])))
            offset += n
        slabs[slot] = views
    return views


def mp_cpu_tier_store(local_context: dict, cache_id: int, slot: int, page_index: int):
    ts = _mp_cpu_tier_tensors(local_context, cache_id)
    for v, t in zip(_mp_cpu_tier_views(local_context, cache_id, slot), ts):
        v.copy_(t[page_index], non_blocking = True)


def mp_cpu_tier_fetch(local_context: dict, cache_id: int, slot: int, page_index: int):
    ts = _mp_cpu_tier_tensors(local_context, cache_id)
    for v, t in zip(_mp_cpu_tier_views(local_context, cache_id, slot), ts):
        t[page_index].copy_(v, non_blocking = True)


class TPCPUPageCache(CPUPageCache):

    def __init__(self, caches: list, max_size: int):
        tp_caches = [c for c in caches if c.model.loaded_tp]
        local_caches = [c for c in caches if not c.model.loaded_tp]
        assert len(tp_caches) == 1, "TPCPUPageCache expects exactly one tensor-parallel cache"
        self.tp_cache = tp_caches[0]
        self.tp_model = self.tp_cache.model
        self.tp_cache_id = id(self.tp_cache)

        # Bytes per page held by the workers (sum over TP shards)
        m = self.tp_model
        for device in m.active_devices:
            m.tp_worker_dispatch(device, mp_cpu_tier_page_bytes, (self.tp_cache_id,))
        self.tp_page_bytes = sum(m.tp_worker_result(device) for device in m.active_devices)
        assert self.tp_page_bytes > 0, "No TP cache tensors found for CPU page cache tier"

        # Local (non-TP) part: base-class slabs, possibly empty
        self.segments = []
        offset = 0
        for cache in local_caches:
            for layer in cache.layers.values():
                for t in layer.get_tensors():
                    if t is None:
                        continue
                    self.segments.append((t, offset, tuple(t.shape[1:]), t.dtype))
                    offset = _align(offset + t[0].numel() * t.element_size(), 256)
        self.slot_size = _align(max(offset, 1), 4096)
        self.max_slots = int(max_size) // (self.slot_size + self.tp_page_bytes)
        assert self.max_slots >= 2, "CPU page cache too small for two pages"

        self.pagetable = None
        self.entries = {}
        self.slot_slabs = []
        self.slot_views = []
        self.free_slots = deque()
        self._order = deque()
        self._order_pops = 0
        self._order_rebuild = max(64, self.max_slots // 8)
        self.metrics = {"pushes": 0, "dedup_hits": 0, "restores": 0, "evictions": 0, "cold_allocs": 0}
        # Worker slabs are pinned lazily on first use of a slot; the local part keeps the background pinner
        self._spare = deque()
        self._spare_cond = threading.Condition()
        self._alloc_thread = threading.Thread(target = self._alloc_worker, daemon = True)
        self._alloc_thread.start()

    def _dispatch(self, fn, slot: int, page_index: int):
        m = self.tp_model
        for device in m.active_devices:
            m.tp_worker_dispatch(device, fn, (self.tp_cache_id, slot, page_index))
        for device in m.active_devices:
            m.tp_worker_result(device)

    def store(self, page, serial: int, protect: set | None = None):
        e = self.entries.get(page.phash)
        if e is not None:
            e["access_serial"] = serial
            self.metrics["dedup_hits"] += 1
            return
        slot = self._new_slot(protect)
        for v, (t, _, _, _) in zip(self.slot_views[slot], self.segments):
            v.copy_(t[page.page_index], non_blocking = True)
        self._dispatch(mp_cpu_tier_store, slot, page.page_index)
        self.entries[page.phash] = {
            "slot": slot,
            "prev_hash": page.prev_hash,
            "access_serial": serial,
            "tokens": page.sequence.clone(),
        }
        self.metrics["pushes"] += 1

    def fetch(self, phash: bytes, page_index: int, serial: int) -> dict:
        e = self.entries[phash]
        e["access_serial"] = serial
        for v, (t, _, _, _) in zip(self.slot_views[e["slot"]], self.segments):
            t[page_index].copy_(v, non_blocking = True)
        self._dispatch(mp_cpu_tier_fetch, e["slot"], page_index)
        self.metrics["restores"] += 1
        return e
