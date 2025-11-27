import pandas as pd
from datetime import datetime, date


def create_sample_microbiology_data() -> pd.DataFrame:
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


def create_sample_transfers_data() -> pd.DataFrame:
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


def create_sample_contacts() -> list[dict]:
    """Create sample contact data for testing."""
    return [
        {
            "patient1": "P001",
            "patient2": "P002", 
            "location": "Ward-1",
            "contact_date": date(2025, 1, 2)
        },
        {
            "patient1": "P002",
            "patient2": "P003",
            "location": "Ward-2", 
            "contact_date": date(2025, 1, 3)
        }
    ]