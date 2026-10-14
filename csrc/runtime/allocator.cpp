/*
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in Lean's LICENSE.
Author: Julia M. Himmel

Adapted from Lean 4.35's runtime/mimalloc.cpp and runtime/alloc_tls.h.
Keep the allocation entry points in the same translation unit as mimalloc:
otherwise the static runtime pulls in its own allocator as well.
*/
#include <static.c>
#undef _Atomic
#include <lean/lean.h>

extern "C" {
struct lean_runtime_tls {
    uint64_t heartbeat;
    mi_theap_s * mi_theap_default;
};
#ifdef _MSC_VER
__declspec(thread) lean_runtime_tls lean_g_tls = {0, (mi_theap_t*)&_mi_theap_empty};
#else
__thread lean_runtime_tls lean_g_tls = {0, (mi_theap_t*)&_mi_theap_empty};
#endif

void lean_mi_theap_cache_init(void) {
    lean_g_tls.mi_theap_default = mi_theap_get_default();
}

LEAN_EXPORT LEAN_ATTR_MALLOC lean_object * lean_alloc_small_object_core(unsigned sz) {
    lean_runtime_tls * tls = &lean_g_tls;
    tls->heartbeat++;
    assert(sz > 0 && sz % LEAN_OBJECT_SIZE_DELTA == 0 && sz <= MI_SMALL_SIZE_MAX);
    void * mem = mi_theap_malloc_small(tls->mi_theap_default, sz);
    if (LEAN_UNLIKELY(mem == NULL)) lean_internal_panic_out_of_memory();
    lean_object * object = (lean_object*)mem;
    // Lean's object-size accounting needs the requested size, not the allocation capacity.
    object->m_cs_sz = sz;
    return object;
}
}
