"""
Pytest configuration and shared fixtures for the NEX Backend test suite.

This file provides centralized test configuration and reusable fixtures
for testing the infection detection and contact tracing system.
"""

import asyncio
import os
import tempfile
from datetime import datetime, date
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.infrastructure.services.infection_detection_service import (
    InfectionDetectionService,
    infection_detection_service
)


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_microbiology_data() -> pd.DataFrame:
    """Create sample microbiology test data for testing."""
    data = [
        {
            "test_id": "T001",
            "patient_id": "P001",
            "collection_date": datetime(2025, 1, 1),
            "organism": "CRE",
            "result": "positive"
        },
        {
            "test_id": "T002", 
            "patient_id": "P002",
            "collection_date": datetime(2025, 1, 2),
            "organism": "CRE",
            "result": "positive"
        },
        {
            "test_id": "T003",
            "patient_id": "P003",
            "collection_date": datetime(2025, 1, 3),
            "organism": "MRSA",
            "result": "positive"
        },
        {
            "test_id": "T004",
            "patient_id": "P004",
            "collection_date": datetime(2025, 1, 4),
            "organism": "CRE",
            "result": "negative"
        },
        {
            "test_id": "T005",
            "patient_id": "P005",
            "collection_date": datetime(2025, 1, 5),
            "organism": "ESBL",
            "result": "positive"
        }
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_transfers_data() -> pd.DataFrame:
    """Create sample transfer data for testing."""
    data = [
        {
            "transfer_id": "TR001",
            "patient_id": "P001",
            "ward_in_time": datetime(2025, 1, 1),
            "ward_out_time": datetime(2025, 1, 3),
            "location": "Ward-1"
        },
        {
            "transfer_id": "TR002",
            "patient_id": "P002", 
            "ward_in_time": datetime(2025, 1, 2),
            "ward_out_time": datetime(2025, 1, 4),
            "location": "Ward-1"
        },
        {
            "transfer_id": "TR003",
            "patient_id": "P003",
            "ward_in_time": datetime(2025, 1, 3),
            "ward_out_time": datetime(2025, 1, 5),
            "location": "Ward-2"
        },
        {
            "transfer_id": "TR004",
            "patient_id": "P005",
            "ward_in_time": datetime(2025, 1, 1),
            "ward_out_time": datetime(2025, 1, 2),
            "location": "Ward-3"
        }
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_contacts() -> list[dict]:
    """Create sample contact data for testing."""
    return [
        {
            "patient1": "P001",
            "patient2": "P002", 
            "location": "Ward-1",
            "contact_date": date(2025, 1, 2),
            "days_from_test1": 1,
            "days_from_test2": 0
        },
        {
            "patient1": "P002",
            "patient2": "P003",
            "location": "Ward-2", 
            "contact_date": date(2025, 1, 3),
            "days_from_test1": 1,
            "days_from_test2": 0
        }
    ]


@pytest.fixture
def large_dataset_microbiology() -> pd.DataFrame:
    """Create a large microbiology dataset for performance testing."""
    import random
    
    organisms = ["CRE", "MRSA", "VRE", "ESBL", "MSSA"]
    results = ["positive", "negative"]
    
    data = []
    for i in range(1000):
        data.append({
            "test_id": f"T{i:04d}",
            "patient_id": f"P{i%200:03d}",  # 200 unique patients
            "collection_date": datetime(2025, 1, 1) + pd.Timedelta(days=i%90),
            "organism": random.choice(organisms),
            "result": random.choice(results)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def large_dataset_transfers() -> pd.DataFrame:
    """Create a large transfers dataset for performance testing."""
    import random
    
    locations = [f"Ward-{i}" for i in range(1, 21)]  # 20 wards
    
    data = []
    for i in range(1500):
        start_date = datetime(2025, 1, 1) + pd.Timedelta(days=i%90)
        end_date = start_date + pd.Timedelta(days=random.randint(1, 10))
        
        data.append({
            "transfer_id": f"TR{i:04d}",
            "patient_id": f"P{i%200:03d}",
            "ward_in_time": start_date,
            "ward_out_time": end_date,
            "location": random.choice(locations)
        })
    
    return pd.DataFrame(data)


# ============================================================================
# Service and App Fixtures
# ============================================================================

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_service() -> Generator[MagicMock, None, None]:
    """Create a mock infection detection service."""
    with patch('app.infrastructure.services.infection_detection_service.infection_detection_service') as mock:
        # Set up default mock behavior
        mock.contacts = []
        mock.contact_groups = []
        mock.df_micro = None
        mock.df_transfers = None
        mock.df_positive = None
        
        # Mock async methods
        mock.run_detection_pipeline = AsyncMock()
        mock.load_and_optimize_data = AsyncMock()
        
        yield mock


@pytest.fixture
def fresh_service(test_data_dir) -> Generator[InfectionDetectionService, None, None]:
    """Create a fresh infection detection service instance for testing."""
    # Create temporary CSV files
    micro_file = test_data_dir / "test_microbiology.csv"
    transfers_file = test_data_dir / "test_transfers.csv"
    
    service = InfectionDetectionService(
        microbiology_file=str(micro_file),
        transfers_file=str(transfers_file)
    )
    yield service


@pytest.fixture
def service_with_data(fresh_service, sample_microbiology_data, sample_transfers_data, test_data_dir):
    """Create a service instance with sample data loaded."""
    # Write sample data to CSV files
    micro_file = test_data_dir / "test_microbiology.csv"
    transfers_file = test_data_dir / "test_transfers.csv"
    
    sample_microbiology_data.to_csv(micro_file, index=False)
    sample_transfers_data.to_csv(transfers_file, index=False)
    
    # Load data into service
    fresh_service.df_micro = sample_microbiology_data
    fresh_service.df_transfers = sample_transfers_data
    fresh_service.df_positive = sample_microbiology_data[
        sample_microbiology_data['result'] == 'positive'
    ]
    
    yield fresh_service


# ============================================================================
# Mock Response Fixtures
# ============================================================================

@pytest.fixture
def mock_cluster_data():
    """Mock cluster data for testing."""
    return [
        {
            "cluster_id": 1,
            "patients": [
                {"patient_id": "P001", "infections": ["CRE"], "test_dates": ["2025-01-01"]},
                {"patient_id": "P002", "infections": ["CRE"], "test_dates": ["2025-01-02"]}
            ],
            "patient_count": 2,
            "contact_count": 1,
            "infections": ["CRE"],
            "locations": ["Ward-1"],
            "date_range": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-02"
            }
        }
    ]


@pytest.fixture
def mock_graph_data():
    """Mock graph data for testing."""
    return {
        "P001": {
            "contacts": ["P002"],
            "contact_count": 1,
            "infections": ["CRE"],
            "primary_infection": "CRE",
            "test_dates": ["2025-01-01"],
            "locations": ["Ward-1"]
        },
        "P002": {
            "contacts": ["P001"],
            "contact_count": 1,
            "infections": ["CRE"],
            "primary_infection": "CRE", 
            "test_dates": ["2025-01-02"],
            "locations": ["Ward-1"]
        }
    }


@pytest.fixture
def mock_summary_metrics():
    """Mock summary metrics for testing."""
    return {
        "total_patients": 5,
        "connected_patients": 2,
        "isolated_patients": 3,
        "total_clusters": 1,
        "total_contact_events": 1,
        "infection_distribution": {"CRE": 2, "MRSA": 1, "ESBL": 1},
        "location_distribution": {"Ward-1": 2, "Ward-2": 1, "Ward-3": 1},
        "largest_cluster_size": 2
    }


@pytest.fixture
def mock_patient_details():
    """Mock patient details for testing."""
    return {
        "patients": [
            {
                "patient_id": "P001",
                "test_cases": [
                    {
                        "test_id": "T001",
                        "collection_date": "2025-01-01",
                        "organism": "CRE",
                        "result": "positive"
                    }
                ],
                "transfers": [
                    {
                        "transfer_id": "TR001",
                        "location": "Ward-1",
                        "ward_in_time": "2025-01-01 00:00:00",
                        "ward_out_time": "2025-01-03 00:00:00",
                        "duration_hours": 48.0
                    }
                ],
                "positive_infections": ["CRE"],
                "total_tests": 1,
                "total_transfers": 1,
                "first_positive_date": "2025-01-01",
                "last_test_date": "2025-01-01"
            }
        ],
        "total_patients": 1,
        "total_tests": 1,
        "total_transfers": 1
    }


@pytest.fixture 
def mock_visualization_data():
    """Mock visualization data for testing."""
    return {
        "network_nodes": [
            {"id": "P001", "infections": ["CRE"], "primary_infection": "CRE"},
            {"id": "P002", "infections": ["CRE"], "primary_infection": "CRE"}
        ],
        "network_edges": [
            {"source": "P001", "target": "P002", "contact_date": "2025-01-02"}
        ],
        "timeline_events": [
            {
                "date": "2025-01-01",
                "event_type": "positive_test",
                "patient_id": "P001",
                "details": {"organism": "CRE"}
            }
        ],
        "stats": {
            "total_patients": 2,
            "total_contacts": 1,
            "date_range": {"start": "2025-01-01", "end": "2025-01-02"}
        }
    }


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_service_state():
    """Reset the global service state before each test."""
    # Reset the singleton service state
    infection_detection_service.contacts = []
    infection_detection_service.contact_groups = []
    infection_detection_service.df_micro = None
    infection_detection_service.df_transfers = None
    infection_detection_service.df_positive = None
    
    yield
    
    # Clean up after test
    infection_detection_service.contacts = []
    infection_detection_service.contact_groups = []


@pytest.fixture
def mock_csv_files(test_data_dir, sample_microbiology_data, sample_transfers_data):
    """Create temporary CSV files with sample data."""
    micro_file = test_data_dir / "microbiology.csv"
    transfers_file = test_data_dir / "transfers.csv"
    
    sample_microbiology_data.to_csv(micro_file, index=False)
    sample_transfers_data.to_csv(transfers_file, index=False)
    
    # Patch the file paths in the service
    with patch.object(infection_detection_service, 'microbiology_file', str(micro_file)):
        with patch.object(infection_detection_service, 'transfers_file', str(transfers_file)):
            yield {
                "microbiology_file": str(micro_file),
                "transfers_file": str(transfers_file)
            }


@pytest.fixture
def async_mock():
    """Create an AsyncMock for async method testing."""
    return AsyncMock()


# ============================================================================
# Performance Testing Fixtures  
# ============================================================================

@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# ============================================================================
# Error Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_file_not_found():
    """Mock file not found error for testing."""
    return FileNotFoundError("Test data files not found")


@pytest.fixture
def mock_invalid_data():
    """Mock invalid data for error testing."""
    return pd.DataFrame({
        "invalid_column": ["invalid", "data"],
        "another_invalid": [1, 2]
    })