import json
import pytest
from jsonschema import validate, ValidationError
from pathlib import Path

# Load schemas
SCHEMA_DIR = Path(__file__).parent.parent / "schemas"

with open(SCHEMA_DIR / "events.schema.json") as f:
    EVENTS_SCHEMA = json.load(f)

with open(SCHEMA_DIR / "state.schema.json") as f:
    STATE_SCHEMA = json.load(f)


def test_events_schema_valid():
    """Test valid event samples"""
    valid_events = [
        {
            "ts": "2026-07-06T09:15:00.123-03:00",
            "level": "info",
            "run_id": "20260706_091500",
            "event": "init",
            "data": {"backend": "simulator"}
        },
        {
            "ts": "2026-07-06T09:16:00.456-03:00",
            "level": "debug",
            "run_id": "20260706_091600",
            "event": "iter",
            "ansatz": "Base",
            "iter": 0,
            "data": {"loss": 0.5, "theta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
        },
        {
            "ts": "2026-07-06T09:17:00.789-03:00",
            "level": "warning",
            "run_id": "20260706_091700",
            "event": "ansatz_start",
            "ansatz": "HEA",
            "data": {"parameters": 4}
        },
        {
            "ts": "2026-07-06T09:18:00.000-03:00",
            "level": "info",
            "run_id": "20260706_091800",
            "event": "calibration_done",
            "data": {"calibration_time": 12.5}
        },
        {
            "ts": "2026-07-06T09:19:00.000-03:00",
            "level": "critical",
            "run_id": "20260706_091900",
            "event": "shutdown",
            "data": {"reason": "completed"}
        },
        {
            "ts": "2026-07-06T09:20:00.000-03:00",
            "level": "info",
            "run_id": "20260706_092000",
            "event": "circuit_exec",
            "ansatz": "Base",
            "iter": 3,
            "data": {"shots": 1024, "counts": {"00": 512, "11": 512}}
        }
    ]
    
    for event in valid_events:
        validate(instance=event, schema=EVENTS_SCHEMA)


def test_events_schema_invalid():
    """Test invalid events that should fail validation"""
    invalid_events = [
        # Missing required field 'ts'
        {
            "level": "info",
            "run_id": "20260706_091500",
            "event": "init"
        },
        # Wrong level enum
        {
            "ts": "2026-07-06T09:15:00.123-03:00",
            "level": "invalid_level",
            "run_id": "20260706_091500",
            "event": "init"
        },
        # Wrong event enum
        {
            "ts": "2026-07-06T09:15:00.123-03:00",
            "level": "info",
            "run_id": "20260706_091500",
            "event": "invalid_event"
        },
        # Malformed run_id
        {
            "ts": "2026-07-06T09:15:00.123-03:00",
            "level": "info",
            "run_id": "invalid_id",
            "event": "init"
        },
        # Malformed timestamp
        {
            "ts": "2026-07-06T09:15:00-03:00",  # Missing milliseconds
            "level": "info",
            "run_id": "20260706_091500",
            "event": "init"
        }
    ]
    
    for event in invalid_events:
        with pytest.raises(ValidationError):
            validate(instance=event, schema=EVENTS_SCHEMA)


def test_state_schema_valid_running():
    """Test valid state.json with status=running"""
    valid_state = {
        "schema_version": 1,
        "run_id": "20260706_091500",
        "status": "running",
        "created_at": "2026-07-06T09:15:00.123-03:00",
        "updated_at": "2026-07-06T09:15:00.123-03:00",
        "config": {
            "backend_requested": "simulator",
            "max_iter": 100,
            "n_shots": 1024,
            "random_state": 42,
            "ansatze": ["Base", "HEA"],
            "optimizers": ["COBYLA"]
        },
        "hardware": {
            "spinqit_version": "1.0.0",
            "shots_per_sec": 1000.0,
            "native_gate_set": ["rx", "ry", "rz", "cx"]
        },
        "phases": {
            "phase1": {
                "status": "completed",
                "started_at": "2026-07-06T09:15:00.123-03:00",
                "ended_at": "2026-07-06T09:20:00.123-03:00",
                "duration_sec": 300.0,
                "ansatze": ["Base"]
            }
        },
        "results": {
            "Base": {
                "metrics": {"accuracy": 0.95},
                "loss_history_path": "/path/to/loss.csv",
                "theta_path": "/path/to/theta.csv",
                "eval_metrics": {"f1_score": 0.92}
            }
        },
        "errors": []
    }
    
    validate(instance=valid_state, schema=STATE_SCHEMA)


def test_state_schema_valid_interrupted():
    """Test valid state.json with status=interrupted (must include interrupted_at_* fields)"""
    valid_state = {
        "schema_version": 1,
        "run_id": "20260706_091500",
        "status": "interrupted",
        "interrupted_at_phase": "phase1",
        "interrupted_at_ansatz": "Base",
        "interrupted_at_iter": 50,
        "created_at": "2026-07-06T09:15:00.123-03:00",
        "updated_at": "2026-07-06T09:25:00.123-03:00",
        "config": {
            "backend_requested": "simulator",
            "max_iter": 100,
            "n_shots": 1024,
            "random_state": 42,
            "ansatze": ["Base"],
            "optimizers": ["COBYLA"]
        },
        "hardware": {
            "spinqit_version": "1.0.0"
        },
        "phases": {
            "phase1": {
                "status": "interrupted",
                "started_at": "2026-07-06T09:15:00.123-03:00",
                "ended_at": "2026-07-06T09:25:00.123-03:00",
                "duration_sec": 600.0,
                "ansatze": ["Base"]
            }
        },
        "results": {
            "Base": {
                "metrics": {"accuracy": 0.85},
                "loss_history_path": "/path/to/loss.csv",
                "theta_path": "/path/to/theta.csv"
            }
        },
        "errors": [
            {
                "ts": "2026-07-06T09:30:00.000-03:00",
                "where": "ansatz_optimization",
                "what": "Timeout exceeded",
                "traceback": "TimeoutError: Iteration limit reached"
            }
        ]
    }
    
    validate(instance=valid_state, schema=STATE_SCHEMA)


def test_state_schema_invalid():
    """Test invalid state.json that should fail validation"""
    invalid_states = [
        # Missing required field 'status'
        {
            "schema_version": 1,
            "run_id": "20260706_091500",
            "created_at": "2026-07-06T09:15:00.123-03:00",
            "config": {
                "backend_requested": "simulator",
                "max_iter": 100,
                "n_shots": 1024,
                "random_state": 42,
                "ansatze": ["Base"],
                "optimizers": ["COBYLA"]
            }
        },
        # Invalid status
        {
            "schema_version": 1,
            "run_id": "20260706_091500",
            "status": "invalid_status",
            "created_at": "2026-07-06T09:15:00.123-03:00",
            "updated_at": "2026-07-06T09:15:00.123-03:00",
            "config": {
                "backend_requested": "simulator",
                "max_iter": 100,
                "n_shots": 1024,
                "random_state": 42,
                "ansatze": ["Base"],
                "optimizers": ["COBYLA"]
            }
        },
        # Missing required config fields
        {
            "schema_version": 1,
            "run_id": "20260706_091500",
            "status": "running",
            "created_at": "2026-07-06T09:15:00.123-03:00",
            "updated_at": "2026-07-06T09:15:00.123-03:00",
            "config": {
                "backend_requested": "simulator"
                # Missing max_iter, n_shots, random_state, ansatze, optimizers
            }
        },
        # Invalid ansatze enum
        {
            "schema_version": 1,
            "run_id": "20260706_091500",
            "status": "running",
            "created_at": "2026-07-06T09:15:00.123-03:00",
            "updated_at": "2026-07-06T09:15:00.123-03:00",
            "config": {
                "backend_requested": "simulator",
                "max_iter": 100,
                "n_shots": 1024,
                "random_state": 42,
                "ansatze": ["InvalidAnsatz"],  # Invalid enum value
                "optimizers": ["COBYLA"]
            }
        }
    ]
    
    for state in invalid_states:
        with pytest.raises(ValidationError):
            validate(instance=state, schema=STATE_SCHEMA)


def test_schema_files_are_valid_json():
    """Test that schema files are valid JSON"""
    assert EVENTS_SCHEMA is not None
    assert STATE_SCHEMA is not None
    
    # Check required top-level keys
    assert "$schema" in EVENTS_SCHEMA
    assert "$id" in EVENTS_SCHEMA
    assert "title" in EVENTS_SCHEMA
    assert "type" in EVENTS_SCHEMA
    
    assert "$schema" in STATE_SCHEMA
    assert "$id" in STATE_SCHEMA
    assert "title" in STATE_SCHEMA
    assert "type" in STATE_SCHEMA