# VecDB Orchestration Guide

> **Companion to:** `PRD.md`
> **Purpose:** Task assignments, dependencies, and workflow for agentic development

---

## Development Strategy: Two-Phase Approach

This project uses a **two-phase development strategy** to reduce integration risk and enable faster iteration.

| Phase | Focus | HNSW State | Goal |
|-------|-------|------------|------|
| **Phase 1** | Python modules + Mock HNSW | Brute-force mock (Python) + C++ build skeleton | Validate all Python modules work correctly |
| **Phase 2** | Full C++ HNSW | Real HNSW algorithm | Replace mock with production implementation |

**Rationale:**
- Phase 1 catches API design issues early (before complex C++ work)
- Python-only iteration is faster (no compile cycles)
- Clear fault isolation: if Phase 2 breaks something, it's the C++ code
- Test suite from Phase 1 becomes regression tests for Phase 2

---

## 1. Agents

| Agent ID | Role | Model | Phase 1 Scope | Phase 2 Scope |
|----------|------|-------|---------------|---------------|
| `orchestrator` | Project coordinator, code review | **Opus** | All reviews | All reviews |
| `hnsw_specialist` | C++ HNSW implementation | **Opus** | CMake skeleton, trivial binding | Full HNSW implementation |
| `python_dev` | Python modules | **Sonnet** | All PY-* tasks, Mock HNSW | Minor fixes if needed |
| `devops` | Build system, testing, docs | **Sonnet** | SETUP-*, Phase 1 integration | Phase 2 integration, benchmarks, docs |

---

## 2. Task Registry

### Phase 1: Python-Focused Development

| Task ID | Task Name | Agent | Depends On |
|---------|-----------|-------|------------|
| SETUP-001 | Project skeleton | devops | — |
| SETUP-002 | Mock HNSW (brute-force) | python_dev | SETUP-001 |
| SETUP-003 | C++ build skeleton | hnsw_specialist | SETUP-001 |
| PY-001 | Exception classes | python_dev | SETUP-001 |
| PY-002 | Collection Manager | python_dev | SETUP-002, PY-001 |
| PY-003 | Persistence Manager | python_dev | PY-001 |
| PY-004 | API Layer (VecDB class) | python_dev | PY-002, PY-003 |
| PY-005 | Python unit tests | python_dev | PY-002, PY-003, PY-004 |
| INT-P1-001 | Phase 1 integration tests | devops | PY-005, SETUP-003 |
| REVIEW-P1 | Phase 1 review & approval | orchestrator | INT-P1-001 |

### Phase 2: C++ HNSW Implementation

| Task ID | Task Name | Agent | Depends On |
|---------|-----------|-------|------------|
| HNSW-001 | HNSW core implementation | hnsw_specialist | REVIEW-P1 |
| HNSW-002 | Distance functions | hnsw_specialist | HNSW-001 |
| HNSW-003 | pybind11 bindings | hnsw_specialist | HNSW-001, HNSW-002 |
| HNSW-004 | HNSW unit tests | hnsw_specialist | HNSW-003 |
| INT-P2-001 | Phase 2 integration build | devops | HNSW-004 |
| INT-P2-002 | Phase 2 integration tests | devops | INT-P2-001 |
| INT-P2-003 | Benchmarks | devops | INT-P2-001 |
| DOC-001 | README and documentation | devops | INT-P2-002 |
| FINAL-001 | Final review | orchestrator | DOC-001 |

---

## 3. Task Details

---

## PHASE 1: Python-Focused Development

> **Goal:** Complete, tested Python modules using a mock HNSW. Validates API design before C++ investment.

---

### SETUP-001: Project Skeleton

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | — |
| **PRD Reference** | Section 6.4, Section 8 |

**Status:** ✅ COMPLETE (container environment ready)

**Output:**
```
vecdb/
├── .devcontainer/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
├── src/
│   ├── cpp/
│   └── python/vecdb/
└── tests/
```

**Acceptance Criteria:**
- ✅ Directory structure exists
- ✅ Docker container builds and runs
- ✅ `pytest tests/` executes successfully

---

### SETUP-002: Mock HNSW (Brute-Force)

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | SETUP-001 |
| **PRD Reference** | Section 3.1.2 (interface spec) |

**Input:** Project skeleton from SETUP-001

**Output:**
```
src/python/vecdb/
└── _hnsw_mock.py    # Functional mock with brute-force search
```

**Acceptance Criteria:**
- Implements `HNSWIndex` class matching PRD 3.1.2 interface **exactly**
- Uses brute-force O(N) search internally (no graph structure needed)
- All distance metrics implemented (L2, cosine, dot product)
- Serialization returns/accepts bytes (can use pickle for mock)
- Raises correct exceptions: `DimensionError`, `DuplicateIDError`
- Works with 100+ vectors for testing

**Mock Implementation Guidance:**
```python
class HNSWIndex:
    """Mock HNSW using brute-force search. Matches real interface exactly."""

    def __init__(self, dimension: int, metric: str, M: int = 16, ef_construction: int = 200):
        self.dimension = dimension
        self.metric = metric
        self.M = M  # Stored but unused in mock
        self.ef_construction = ef_construction  # Stored but unused in mock
        self._vectors: dict[int, list[float]] = {}
        self._deleted: set[int] = set()

    def add(self, internal_id: int, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise DimensionError(f"Expected {self.dimension}, got {len(vector)}")
        if internal_id in self._vectors and internal_id not in self._deleted:
            raise DuplicateIDError(f"ID {internal_id} already exists")
        self._vectors[internal_id] = vector
        self._deleted.discard(internal_id)

    def search(self, query: list[float], k: int, ef_search: int = 50) -> list[tuple[int, float]]:
        if len(query) != self.dimension:
            raise DimensionError(f"Expected {self.dimension}, got {len(query)}")

        # Brute-force: compute all distances
        results = []
        for id, vec in self._vectors.items():
            if id not in self._deleted:
                dist = self._compute_distance(query, vec)
                results.append((id, dist))

        results.sort(key=lambda x: x[1])
        return results[:k]

    def remove(self, internal_id: int) -> bool:
        if internal_id in self._vectors and internal_id not in self._deleted:
            self._deleted.add(internal_id)
            return True
        return False

    def serialize(self) -> bytes:
        import pickle
        return pickle.dumps({
            'dimension': self.dimension,
            'metric': self.metric,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'vectors': self._vectors,
            'deleted': self._deleted,
        })

    @staticmethod
    def deserialize(data: bytes) -> 'HNSWIndex':
        import pickle
        state = pickle.loads(data)
        index = HNSWIndex(state['dimension'], state['metric'], state['M'], state['ef_construction'])
        index._vectors = state['vectors']
        index._deleted = state['deleted']
        return index

    def __len__(self) -> int:
        return len(self._vectors) - len(self._deleted)

    def _compute_distance(self, a: list[float], b: list[float]) -> float:
        # Implement based on self.metric
        ...
```

---

### SETUP-003: C++ Build Skeleton

| Field | Value |
|-------|-------|
| **Agent** | `hnsw_specialist` |
| **Depends on** | SETUP-001 |
| **PRD Reference** | Section 6 |

**Input:** Project skeleton

**Output:**
```
src/cpp/
├── CMakeLists.txt      # Build configuration with pybind11
└── stub.cpp            # Minimal pybind11 module (compiles & imports)

CMakeLists.txt          # Top-level CMake (updated)
```

**Acceptance Criteria:**
- `cd build && cmake .. && make` completes without errors
- `python -c "import vecdb._hnsw_cpp"` succeeds (even if module is empty)
- CMake correctly finds pybind11
- Build produces `_hnsw_cpp.cpython-*.so`

**Minimal stub.cpp:**
```cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_hnsw_cpp, m) {
    m.doc() = "HNSW C++ module - stub for Phase 1, full implementation in Phase 2";

    // Placeholder: real HNSWIndex class will be added in Phase 2
    m.def("is_stub", []() { return true; }, "Returns true if this is the stub module");
}
```

**Purpose:** Validates the C++/pybind11/CMake pipeline works before Phase 2.

---

### PY-001: Exception Classes

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | SETUP-001 |
| **PRD Reference** | Section 4.3 |

**Output:**
```
src/python/vecdb/exceptions.py
```

**Acceptance Criteria:**
- Defines: `VecDBError` (base), `DimensionError`, `DuplicateIDError`, `CollectionExistsError`, `CollectionNotFoundError`, `DeserializationError`
- All inherit from `VecDBError`
- Each has descriptive default message
- Can be imported: `from vecdb.exceptions import DimensionError`

---

### PY-002: Collection Manager

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | SETUP-002, PY-001 |
| **PRD Reference** | Section 3.2 |

**Input:**
- `_hnsw_mock.py` from SETUP-002
- `exceptions.py` from PY-001

**Output:**
```
src/python/vecdb/collection.py
```

**Acceptance Criteria:**
- Implements `Collection` class per PRD 3.2.2
- Maintains `user_to_internal` and `internal_to_user` mappings
- Stores original vectors in `vectors` dict for `get()` retrieval
- Validates dimension on `insert()` before calling HNSW
- Translates internal IDs to user IDs in `search()` results
- Raises `DuplicateIDError` on duplicate user ID insert
- Type hints on all public methods
- Uses mock HNSW from `_hnsw_mock.py`

**Import Strategy:**
```python
# In collection.py - allows swapping mock for real later
try:
    from vecdb._hnsw_cpp import HNSWIndex  # Phase 2: real C++
except ImportError:
    from vecdb._hnsw_mock import HNSWIndex  # Phase 1: mock
```

---

### PY-003: Persistence Manager

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | PY-001 |
| **PRD Reference** | Section 3.3 |

**Output:**
```
src/python/vecdb/persistence.py
```

**Acceptance Criteria:**
- Implements `PersistenceManager` class per PRD 3.3.2
- Creates directory structure: `db_path/collections/`
- Saves `.meta` (JSON), `.hnsw` (binary), `.vectors` (binary) per collection
- Uses atomic writes (write to `.tmp`, then `os.rename`)
- Includes version number in file headers
- `load_collection()` returns `None` if not found (no exception)
- Handles corrupt files gracefully with `DeserializationError`

---

### PY-004: API Layer (VecDB Class)

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | PY-002, PY-003 |
| **PRD Reference** | Section 3.4 |

**Output:**
```
src/python/vecdb/vecdb.py
src/python/vecdb/__init__.py (updated with exports)
```

**Acceptance Criteria:**
- Implements `VecDB` class per PRD 3.4.2
- Constructor loads existing collections from disk
- `create_collection()` raises `CollectionExistsError` if name taken
- `get_collection()` raises `CollectionNotFoundError` if missing
- `save()` persists all collections
- Context manager (`__enter__`/`__exit__`) auto-saves on exit
- `__init__.py` exports: `VecDB`, `Collection`, all exceptions

---

### PY-005: Python Unit Tests

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | PY-002, PY-003, PY-004 |
| **PRD Reference** | Section 7.2.2, Section 7.2.3 |

**Output:**
```
tests/python/
├── test_hnsw_mock.py       # Tests for mock HNSW
├── test_collection.py      # Collection Manager tests
├── test_persistence.py     # Persistence tests
└── test_vecdb.py           # API layer tests
```

**Acceptance Criteria:**
- `test_hnsw_mock.py`: All distance metrics correct, add/search/remove work
- `test_collection.py`: All tests from PRD 7.2.2
- `test_persistence.py`: All tests from PRD 7.2.3
- `test_vecdb.py`: Unit tests for VecDB class
- All tests pass with `pytest tests/python/`

---

### INT-P1-001: Phase 1 Integration Tests

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | PY-005, SETUP-003 |
| **PRD Reference** | Section 7.2.4 |

**Output:**
```
tests/python/test_integration.py
```

**Acceptance Criteria:**
- `test_full_workflow`: create → insert → search → save → reload → search
- `test_multiple_collections`: two collections, different dimensions
- `test_context_manager`: `with VecDB(...) as db:` auto-saves
- All tests pass using **mock HNSW**
- C++ stub compiles (validates build pipeline)

---

### REVIEW-P1: Phase 1 Review & Approval

| Field | Value |
|-------|-------|
| **Agent** | `orchestrator` |
| **Depends on** | INT-P1-001 |

**Acceptance Criteria:**
- All Python modules implemented per PRD specs
- All unit tests passing
- All integration tests passing (with mock)
- Code quality acceptable
- Interface matches PRD 3.1.2 exactly (critical for Phase 2)

**Gate:** Phase 2 cannot begin until REVIEW-P1 is approved.

---

## PHASE 2: C++ HNSW Implementation

> **Goal:** Replace mock with production HNSW. Python modules should require minimal changes.

---

### HNSW-001: HNSW Core Implementation

| Field | Value |
|-------|-------|
| **Agent** | `hnsw_specialist` |
| **Depends on** | REVIEW-P1 |
| **PRD Reference** | Section 3.1, Section 5.1 |

**Input:**
- Approved Phase 1 codebase
- Interface spec from `_hnsw_mock.py` (must match exactly)

**Output:**
```
src/cpp/
├── hnsw_index.hpp
└── hnsw_index.cpp
```

**Acceptance Criteria:**
- Implements HNSW graph structure with multi-layer navigation
- Implements `add()` with level assignment via exponential distribution
- Implements `search()` with greedy layer-by-layer descent
- Implements `remove()` with lazy deletion (mark, don't rebuild)
- Implements `serialize()` / `deserialize()` for binary persistence
- Compiles with C++17, no warnings with `-Wall -Wextra`
- No dependencies beyond C++ standard library

---

### HNSW-002: Distance Functions

| Field | Value |
|-------|-------|
| **Agent** | `hnsw_specialist` |
| **Depends on** | HNSW-001 |
| **PRD Reference** | Section 10 (Appendix) |

**Output:**
```
src/cpp/
├── distance.hpp
└── distance.cpp
```

**Acceptance Criteria:**
- Implements `l2_distance(a, b, dim)` — Euclidean distance
- Implements `cosine_distance(a, b, dim)` — returns `1 - cosine_similarity`
- Implements `dot_distance(a, b, dim)` — returns `-dot_product` (for max inner product)
- Functions accept `const float*` pointers and `size_t` dimension
- Results match mock implementation exactly (within floating-point tolerance)

---

### HNSW-003: pybind11 Bindings

| Field | Value |
|-------|-------|
| **Agent** | `hnsw_specialist` |
| **Depends on** | HNSW-001, HNSW-002 |
| **PRD Reference** | Section 3.1.2, Section 4.2 |

**Output:**
```
src/cpp/
├── bindings.cpp
└── CMakeLists.txt (updated)
```

**Acceptance Criteria:**
- Python can `from vecdb._hnsw_cpp import HNSWIndex`
- **Interface matches `_hnsw_mock.py` exactly** (critical!)
- Accepts `list[float]` or `numpy.ndarray` for vectors
- Returns `list[tuple[int, float]]` for search results
- C++ exceptions translate to Python exceptions
- Builds as `_hnsw_cpp.cpython-*.so` shared library

---

### HNSW-004: HNSW Unit Tests

| Field | Value |
|-------|-------|
| **Agent** | `hnsw_specialist` |
| **Depends on** | HNSW-003 |
| **PRD Reference** | Section 7.2.1 |

**Output:**
```
tests/python/test_hnsw_cpp.py
```

**Acceptance Criteria:**
- All tests from PRD 7.2.1:
  - `test_distance_l2`, `test_distance_cosine`, `test_distance_dot`
  - `test_add_single`, `test_add_multiple`
  - `test_search_exact`, `test_search_recall`
  - `test_remove`
  - `test_serialize_deserialize`
  - `test_dimension_mismatch`, `test_duplicate_id`
- Recall test: >95% recall@10 vs brute force on 1000 random vectors

---

### INT-P2-001: Phase 2 Integration Build

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | HNSW-004 |
| **PRD Reference** | Section 6.4 |

**Acceptance Criteria:**
- C++ module builds in container
- `python -c "from vecdb._hnsw_cpp import HNSWIndex"` succeeds
- Python modules automatically use C++ instead of mock

**Integration Switch:**
```python
# collection.py should now import real C++ module
try:
    from vecdb._hnsw_cpp import HNSWIndex  # This should succeed now
except ImportError:
    from vecdb._hnsw_mock import HNSWIndex  # Fallback (shouldn't hit)
```

---

### INT-P2-002: Phase 2 Integration Tests

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | INT-P2-001 |
| **PRD Reference** | Section 7.2.4 |

**Acceptance Criteria:**
- All Phase 1 integration tests still pass (regression)
- Tests now use real C++ HNSW (not mock)
- Performance is reasonable (search < 100ms for 10K vectors)

---

### INT-P2-003: Benchmarks

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | INT-P2-001 |
| **PRD Reference** | Section 4 (Deliverables) |

**Output:**
```
benchmarks/benchmark_search.py
docs/BENCHMARKS.md
```

**Acceptance Criteria:**
- Benchmark insert throughput (vectors/second) at N=10K, 100K
- Benchmark search latency (ms) at N=10K, 100K for k=10
- Benchmark recall@10 vs brute force
- Compare mock vs C++ performance
- Document results in `BENCHMARKS.md`

---

### DOC-001: README and Documentation

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | INT-P2-002 |
| **PRD Reference** | Section 4 (Deliverables) |

**Output:**
```
README.md (updated)
```

**Acceptance Criteria:**
- Build instructions (Docker and local)
- API reference with code examples
- Design Doc section: architecture, HNSW explanation, persistence strategy
- Trade-offs & Future Improvements section
- Benchmark results summary

---

### FINAL-001: Final Review

| Field | Value |
|-------|-------|
| **Agent** | `orchestrator` |
| **Depends on** | DOC-001 |
| **PRD Reference** | Section 5 (Evaluation Criteria) |

**Acceptance Criteria:**
- All PRD requirements met
- All tests pass
- Documentation complete
- Code quality acceptable

---

## 4. Workflow Diagram

```
═══════════════════════════════════════════════════════════════════════════════
                              PHASE 1: Python Focus
═══════════════════════════════════════════════════════════════════════════════

Day 1: Setup
─────────────────────────────────────────────────────────────────────────────
    [SETUP-001: Project Skeleton] ✅ COMPLETE
           │
           ├────────────────────┬─────────────────────┐
           ▼                    ▼                     ▼
    [SETUP-002: Mock HNSW]  [SETUP-003: C++ Skeleton]  [PY-001: Exceptions]
       python_dev              hnsw_specialist           python_dev

Day 2-4: Python Development
─────────────────────────────────────────────────────────────────────────────

    [SETUP-002] ──────┐
                      │
    [PY-001] ─────────┼──► [PY-002: Collection] ──┐
                      │       python_dev          │
                      │                           ├──► [PY-004: API Layer]
    [PY-001] ─────────┴──► [PY-003: Persistence] ─┘       python_dev
                             python_dev                        │
                                                               ▼
                                                    [PY-005: Unit Tests]
                                                       python_dev

Day 5: Phase 1 Integration
─────────────────────────────────────────────────────────────────────────────

    [PY-005] ──────┐
                   ├──► [INT-P1-001: Integration Tests] ──► [REVIEW-P1]
    [SETUP-003] ───┘           devops                       orchestrator
                                                                 │
                                                                 ▼
                                                         ════════════════
                                                         PHASE 1 GATE
                                                         ════════════════

═══════════════════════════════════════════════════════════════════════════════
                              PHASE 2: C++ Focus
═══════════════════════════════════════════════════════════════════════════════

Day 6-7: C++ HNSW Implementation
─────────────────────────────────────────────────────────────────────────────

    [REVIEW-P1] ──► [HNSW-001: Core] ──► [HNSW-002: Distance] ──► [HNSW-003: Bindings]
                    hnsw_specialist      hnsw_specialist          hnsw_specialist
                                                                        │
                                                                        ▼
                                                              [HNSW-004: Tests]
                                                               hnsw_specialist

Day 8: Phase 2 Integration
─────────────────────────────────────────────────────────────────────────────

    [HNSW-004] ──► [INT-P2-001: Build] ──┬──► [INT-P2-002: Integration Tests]
                        devops           │           devops
                                         │
                                         └──► [INT-P2-003: Benchmarks]
                                                     devops

Day 9: Documentation & Final Review
─────────────────────────────────────────────────────────────────────────────

    [INT-P2-002] ──► [DOC-001: Documentation] ──► [FINAL-001: Final Review]
                           devops                      orchestrator

═══════════════════════════════════════════════════════════════════════════════
```

---

## 5. Handoff Protocol

### Task Completion

```
TASK COMPLETE: {task_id}

Output files:
- path/to/file1.py
- path/to/file2.py

Tests: {passed}/{total} passing

Notes: (any issues, decisions made, or deviations from spec)
```

### Blocked / Questions

```
BLOCKED: {task_id}

Blocker: (description)

Question for orchestrator: (specific question)

Proposed solution: (if any)
```

### Phase Gate Review

```
PHASE REVIEW: Phase 1

Status: APPROVED | NEEDS_CHANGES

Checklist:
- [ ] All Python modules implemented
- [ ] All unit tests passing
- [ ] Integration tests passing with mock
- [ ] Interface matches PRD 3.1.2 exactly
- [ ] Code quality acceptable

Issues: (if any)

Decision: Proceed to Phase 2 | Requires fixes
```

---

## 6. File Ownership

### Phase 1 Files

| Path | Owner Agent | Created By Task |
|------|-------------|-----------------|
| `src/python/vecdb/_hnsw_mock.py` | python_dev | SETUP-002 |
| `src/python/vecdb/exceptions.py` | python_dev | PY-001 |
| `src/python/vecdb/collection.py` | python_dev | PY-002 |
| `src/python/vecdb/persistence.py` | python_dev | PY-003 |
| `src/python/vecdb/vecdb.py` | python_dev | PY-004 |
| `src/python/vecdb/__init__.py` | python_dev | PY-004 |
| `src/cpp/CMakeLists.txt` | hnsw_specialist | SETUP-003 |
| `src/cpp/stub.cpp` | hnsw_specialist | SETUP-003 |
| `tests/python/test_*.py` | python_dev | PY-005 |
| `tests/python/test_integration.py` | devops | INT-P1-001 |

### Phase 2 Files

| Path | Owner Agent | Created By Task |
|------|-------------|-----------------|
| `src/cpp/hnsw_index.hpp` | hnsw_specialist | HNSW-001 |
| `src/cpp/hnsw_index.cpp` | hnsw_specialist | HNSW-001 |
| `src/cpp/distance.hpp` | hnsw_specialist | HNSW-002 |
| `src/cpp/distance.cpp` | hnsw_specialist | HNSW-002 |
| `src/cpp/bindings.cpp` | hnsw_specialist | HNSW-003 |
| `tests/python/test_hnsw_cpp.py` | hnsw_specialist | HNSW-004 |
| `benchmarks/benchmark_search.py` | devops | INT-P2-003 |
| `docs/BENCHMARKS.md` | devops | INT-P2-003 |
| `README.md` | devops | DOC-001 |

---

## 7. Quick Reference

### Phase 1 Task Flow

```
SETUP-001 ──► SETUP-002 ──► PY-002 ──► PY-004 ──► PY-005 ──► INT-P1-001 ──► REVIEW-P1
         └──► SETUP-003 ───────────────────────────────────────┘
         └──► PY-001 ────► PY-002
                     └──► PY-003 ────► PY-004
```

### Phase 2 Task Flow

```
REVIEW-P1 ──► HNSW-001 ──► HNSW-002 ──► HNSW-003 ──► HNSW-004 ──► INT-P2-001 ──► INT-P2-002 ──► DOC-001 ──► FINAL-001
                                                                          └──► INT-P2-003 ───────────┘
```

### Agent Summary

| Agent | Phase 1 Tasks | Phase 2 Tasks |
|-------|---------------|---------------|
| `python_dev` | SETUP-002, PY-001..005 | (minimal fixes if needed) |
| `hnsw_specialist` | SETUP-003 | HNSW-001..004 |
| `devops` | INT-P1-001 | INT-P2-001..003, DOC-001 |
| `orchestrator` | REVIEW-P1 | FINAL-001 |

### Critical Success Factors

1. **Mock interface must match PRD 3.1.2 exactly** — Phase 2 depends on this
2. **Phase 1 gate review** — Don't start C++ until Python is solid
3. **Same tests, different backend** — Phase 1 tests become Phase 2 regression tests

---

*End of Orchestration Guide*
