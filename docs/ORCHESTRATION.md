# VecDB Orchestration Guide

> **Companion to:** `PRD.md`  
> **Purpose:** Task assignments, dependencies, and workflow for agentic development

---

## 1. Agents

| Agent ID | Role | Model | Scope |
|----------|------|-------|-------|
| `orchestrator` | Project coordinator, code review, conflict resolution | **Opus** | All tasks |
| `hnsw_specialist` | C++ HNSW implementation | **Opus** | HNSW-* tasks |
| `python_dev` | Python modules (Collection, Persistence, API) | **Sonnet** | PY-* tasks |
| `devops` | Build system, testing, documentation | **Sonnet** | SETUP-*, INT-*, DOC-* tasks |

---

## 2. Task Registry

| Phase | Task ID | Task Name | Agent | Depends On |
|-------|---------|-----------|-------|------------|
| 0 | SETUP-001 | Project skeleton | devops | — |
| 0 | SETUP-002 | Interface stubs | orchestrator | SETUP-001 |
| 1 | HNSW-001 | HNSW core implementation | hnsw_specialist | SETUP-001 |
| 1 | HNSW-002 | Distance functions | hnsw_specialist | HNSW-001 |
| 1 | HNSW-003 | pybind11 bindings | hnsw_specialist | HNSW-001, HNSW-002 |
| 1 | HNSW-004 | HNSW unit tests | hnsw_specialist | HNSW-003 |
| 1 | PY-001 | Exception classes | python_dev | SETUP-001 |
| 1 | PY-002 | Collection Manager | python_dev | SETUP-002, PY-001 |
| 1 | PY-003 | Persistence Manager | python_dev | PY-001 |
| 1 | PY-004 | API Layer (VecDB class) | python_dev | PY-002, PY-003 |
| 1 | PY-005 | Python unit tests | python_dev | PY-002, PY-003, PY-004 |
| 2 | INT-001 | Integration build | devops | HNSW-004, PY-005 |
| 2 | INT-002 | Integration tests | devops | INT-001 |
| 2 | INT-003 | Benchmarks | devops | INT-001 |
| 3 | DOC-001 | README and docs | devops | INT-002 |
| 3 | FINAL-001 | Final review | orchestrator | DOC-001 |

---

## 3. Task Details

### Phase 0: Setup

#### SETUP-001: Project Skeleton

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | — |
| **PRD Reference** | Section 6.4, Section 8 |

**Input:** None (start from empty directory)

**Output:**
```
vecdb/
├── Dockerfile
├── CMakeLists.txt
├── requirements.txt
├── setup.py
├── src/
│   ├── cpp/
│   │   └── CMakeLists.txt
│   └── python/
│       └── vecdb/
│           └── __init__.py (empty)
└── tests/
    ├── cpp/
    └── python/
```

**Acceptance Criteria:**
- Directory structure matches PRD Section 8
- Dockerfile builds successfully (base image, dependencies installed)
- `cmake ..` runs without error in `src/cpp/`
- `pip install -e .` runs without error (even if package is empty)

---

#### SETUP-002: Interface Stubs

| Field | Value |
|-------|-------|
| **Agent** | `orchestrator` |
| **Depends on** | SETUP-001 |
| **PRD Reference** | Section 3.1.2, Section 4.3 |

**Input:** Project skeleton from SETUP-001

**Output:**
```
src/python/vecdb/
├── _hnsw_stub.py      # Mock HNSWIndex for Python dev to code against
└── exceptions.py      # Complete exception definitions
```

**Acceptance Criteria:**
- `_hnsw_stub.py` contains `HNSWIndex` class matching PRD 3.1.2 interface
- Stub methods raise `NotImplementedError` with descriptive messages
- `exceptions.py` defines all exceptions from PRD 4.3
- Python dev can import and use stubs immediately

**Stub Template:**
```python
class HNSWIndex:
    def __init__(self, dimension: int, metric: str, M: int = 16, ef_construction: int = 200):
        raise NotImplementedError("Stub: HNSW C++ module not yet integrated")
    
    def add(self, internal_id: int, vector: list[float]) -> None:
        raise NotImplementedError("Stub")
    
    # ... etc
```

---

### Phase 1: Parallel Development

#### HNSW-001: HNSW Core Implementation

| Field | Value |
|-------|-------|
| **Agent** | `hnsw_specialist` |
| **Depends on** | SETUP-001 |
| **PRD Reference** | Section 3.1, Section 5.1 |

**Input:**
- `src/cpp/` directory with CMakeLists.txt

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

#### HNSW-002: Distance Functions

| Field | Value |
|-------|-------|
| **Agent** | `hnsw_specialist` |
| **Depends on** | HNSW-001 |
| **PRD Reference** | Section 10 (Appendix) |

**Input:**
- `hnsw_index.hpp/cpp` from HNSW-001

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
- HNSW index uses function pointer or template to select metric at construction

---

#### HNSW-003: pybind11 Bindings

| Field | Value |
|-------|-------|
| **Agent** | `hnsw_specialist` |
| **Depends on** | HNSW-001, HNSW-002 |
| **PRD Reference** | Section 3.1.2, Section 4.2 |

**Input:**
- Complete C++ implementation from HNSW-001, HNSW-002

**Output:**
```
src/cpp/
├── bindings.cpp
└── CMakeLists.txt (updated with pybind11)
```

**Acceptance Criteria:**
- Python can `from vecdb._hnsw import HNSWIndex`
- All methods from PRD 3.1.2 are exposed
- Accepts `list[float]` or `numpy.ndarray` for vectors
- Returns `list[tuple[int, float]]` for search results
- C++ exceptions translate to Python exceptions (DimensionError, DuplicateIDError)
- Builds as `_hnsw.cpython-*.so` shared library

---

#### HNSW-004: HNSW Unit Tests

| Field | Value |
|-------|-------|
| **Agent** | `hnsw_specialist` |
| **Depends on** | HNSW-003 |
| **PRD Reference** | Section 7.2.1 |

**Input:**
- Compiled `_hnsw` module from HNSW-003

**Output:**
```
tests/
├── cpp/
│   └── test_hnsw.cpp        # Optional: C++ level tests
└── python/
    └── test_hnsw_binding.py # Required: Test via Python bindings
```

**Acceptance Criteria:**
- All tests from PRD 7.2.1 implemented and passing:
  - `test_distance_l2`, `test_distance_cosine`, `test_distance_dot`
  - `test_add_single`, `test_add_multiple`
  - `test_search_exact`, `test_search_recall`
  - `test_remove`
  - `test_serialize_deserialize`
  - `test_dimension_mismatch`, `test_duplicate_id`
- Recall test: >95% recall@10 vs brute force on 1000 random vectors

---

#### PY-001: Exception Classes

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | SETUP-001 |
| **PRD Reference** | Section 4.3 |

**Input:**
- `exceptions.py` stub from SETUP-002 (or create from scratch)

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

#### PY-002: Collection Manager

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | SETUP-002, PY-001 |
| **PRD Reference** | Section 3.2 |

**Input:**
- `_hnsw_stub.py` (code against stub; real module integrated later)
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

---

#### PY-003: Persistence Manager

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | PY-001 |
| **PRD Reference** | Section 3.3 |

**Input:**
- `exceptions.py` from PY-001
- Understanding of Collection structure from PRD 3.2

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

#### PY-004: API Layer (VecDB Class)

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | PY-002, PY-003 |
| **PRD Reference** | Section 3.4 |

**Input:**
- `collection.py` from PY-002
- `persistence.py` from PY-003

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

#### PY-005: Python Unit Tests

| Field | Value |
|-------|-------|
| **Agent** | `python_dev` |
| **Depends on** | PY-002, PY-003, PY-004 |
| **PRD Reference** | Section 7.2.2, Section 7.2.3 |

**Input:**
- All Python modules from PY-001 through PY-004
- Uses `_hnsw_stub.py` (mock behavior for unit tests)

**Output:**
```
tests/python/
├── test_collection.py
├── test_persistence.py
└── test_vecdb.py (unit level, not integration)
```

**Acceptance Criteria:**
- All tests from PRD 7.2.2 implemented for Collection
- All tests from PRD 7.2.3 implemented for Persistence
- Tests use mocked HNSW (stub or unittest.mock)
- All tests pass with `pytest tests/python/`

---

### Phase 2: Integration

#### INT-001: Integration Build

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | HNSW-004, PY-005 |
| **PRD Reference** | Section 6.4 |

**Input:**
- All C++ sources from HNSW-*
- All Python sources from PY-*
- Dockerfile from SETUP-001

**Output:**
- Successfully built Docker image
- Compiled `_hnsw` module inside container
- Installable Python package

**Acceptance Criteria:**
- `docker build .` completes without error
- Inside container: `python -c "from vecdb import VecDB"` works
- Inside container: `pytest tests/` runs (may have failures, that's INT-002's job)

---

#### INT-002: Integration Tests

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | INT-001 |
| **PRD Reference** | Section 7.2.4 |

**Input:**
- Working Docker image from INT-001

**Output:**
```
tests/python/test_integration.py
```

**Acceptance Criteria:**
- Implements all tests from PRD 7.2.4:
  - `test_full_workflow`: create → insert → search → save → reload → search
  - `test_multiple_collections`: two collections, different dimensions
  - `test_context_manager`: `with VecDB(...) as db:` auto-saves
- All integration tests pass
- Tests run inside Docker container

---

#### INT-003: Benchmarks

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | INT-001 |
| **PRD Reference** | Section 4 (Deliverables) |

**Input:**
- Working system from INT-001

**Output:**
```
benchmarks/
└── benchmark_search.py

docs/
└── BENCHMARKS.md
```

**Acceptance Criteria:**
- Benchmark insert throughput (vectors/second) at N=10K, 100K
- Benchmark search latency (ms) at N=10K, 100K for k=10
- Benchmark recall@10 vs brute force
- Document results in `BENCHMARKS.md`

---

### Phase 3: Documentation

#### DOC-001: README and Documentation

| Field | Value |
|-------|-------|
| **Agent** | `devops` |
| **Depends on** | INT-002 |
| **PRD Reference** | Section 4 (Deliverables) |

**Input:**
- Complete, tested system
- Benchmark results from INT-003

**Output:**
```
README.md
```

**Acceptance Criteria:**
- Build instructions (Docker and local)
- API reference with code examples (can reference PRD 3.4.3)
- Design Doc section: architecture, HNSW explanation, persistence strategy
- Trade-offs & Future Improvements section
- Benchmark results summary

---

#### FINAL-001: Final Review

| Field | Value |
|-------|-------|
| **Agent** | `orchestrator` |
| **Depends on** | DOC-001 |
| **PRD Reference** | Section 5 (Evaluation Criteria) |

**Input:**
- All deliverables

**Output:**
- Review report
- Final approval or fix requests

**Acceptance Criteria:**
- All PRD requirements met
- All tests pass
- Documentation complete
- Code quality acceptable (no obvious issues)

---

## 4. Workflow Phases

```
Timeline: 9 Days
══════════════════════════════════════════════════════════════

Day 1 ─────────────────────────────────────────────────────────
│
│  [SETUP-001] ──► [SETUP-002]
│     devops        orchestrator
│
Day 2-5 ───────────────────────────────────────────────────────
│
│  ┌─────────────────────────┐  ┌─────────────────────────────┐
│  │ HNSW Track (Opus)       │  │ Python Track (Sonnet)       │
│  │                         │  │                             │
│  │ [HNSW-001]              │  │ [PY-001]                    │
│  │     │                   │  │     │                       │
│  │     ▼                   │  │     ▼                       │
│  │ [HNSW-002]              │  │ [PY-002]    [PY-003]        │
│  │     │                   │  │     │           │           │
│  │     ▼                   │  │     └─────┬─────┘           │
│  │ [HNSW-003]              │  │           ▼                 │
│  │     │                   │  │       [PY-004]              │
│  │     ▼                   │  │           │                 │
│  │ [HNSW-004]              │  │           ▼                 │
│  │                         │  │       [PY-005]              │
│  └─────────────────────────┘  └─────────────────────────────┘
│
Day 6-7 ───────────────────────────────────────────────────────
│
│  [INT-001] ──► [INT-002]
│     │              │
│     └──► [INT-003] │
│              │     │
│              ▼     ▼
Day 8-9 ───────────────────────────────────────────────────────
│
│  [DOC-001] ──► [FINAL-001]
│
══════════════════════════════════════════════════════════════
```

---

## 5. Handoff Protocol

### Task Completion

When an agent completes a task, report:

```
TASK COMPLETE: {task_id}

Output files:
- path/to/file1.py
- path/to/file2.py

Tests: {passed}/{total} passing

Notes: (any issues, decisions made, or deviations from spec)
```

### Blocked / Questions

If blocked or need clarification:

```
BLOCKED: {task_id}

Blocker: (description)

Question for orchestrator: (specific question)

Proposed solution: (if any)
```

### Orchestrator Review Response

```
REVIEW: {task_id}

Status: APPROVED | NEEDS_CHANGES

Issues:
- (list of issues if any)

Required changes:
- (specific changes needed)

Approved to proceed: {next_task_ids}
```

---

## 6. File Ownership

| Path | Owner Agent | Created By Task |
|------|-------------|-----------------|
| `Dockerfile` | devops | SETUP-001 |
| `CMakeLists.txt` | devops | SETUP-001 |
| `setup.py` | devops | SETUP-001 |
| `requirements.txt` | devops | SETUP-001 |
| `src/cpp/*.hpp, *.cpp` | hnsw_specialist | HNSW-001, 002, 003 |
| `src/cpp/CMakeLists.txt` | hnsw_specialist | HNSW-003 |
| `src/python/vecdb/exceptions.py` | python_dev | PY-001 |
| `src/python/vecdb/collection.py` | python_dev | PY-002 |
| `src/python/vecdb/persistence.py` | python_dev | PY-003 |
| `src/python/vecdb/vecdb.py` | python_dev | PY-004 |
| `src/python/vecdb/__init__.py` | python_dev | PY-004 |
| `tests/python/test_*.py` | python_dev | PY-005 |
| `tests/python/test_integration.py` | devops | INT-002 |
| `README.md` | devops | DOC-001 |

---

## 7. Quick Reference

### Agent → Task Mapping

```
orchestrator:    SETUP-002, FINAL-001, (all reviews)
hnsw_specialist: HNSW-001, HNSW-002, HNSW-003, HNSW-004
python_dev:      PY-001, PY-002, PY-003, PY-004, PY-005
devops:          SETUP-001, INT-001, INT-002, INT-003, DOC-001
```

### Critical Path

```
SETUP-001 → HNSW-001 → HNSW-002 → HNSW-003 → HNSW-004 → INT-001 → INT-002 → DOC-001 → FINAL-001
```

### Parallel Work

- HNSW-001..004 runs parallel to PY-001..005
- INT-003 (benchmarks) runs parallel to INT-002 (integration tests)

---

*End of Orchestration Guide*