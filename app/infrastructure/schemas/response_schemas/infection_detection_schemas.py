"""
Response schemas for infection detection API endpoints.
"""

from pydantic import BaseModel, Field


class ContactDetailSchema(BaseModel):
    """Schema for contact detail information."""

    with_patient: str = Field(..., description="Patient ID of the contact")
    location: str = Field(..., description="Location where contact occurred")
    date: str = Field(..., description="Date of contact (YYYY-MM-DD)")


class PatientInfoSchema(BaseModel):
    """Schema for patient information in the graph data."""

    contacts: list[str] = Field(
        ..., description="List of patient IDs this patient contacted"
    )
    contact_count: int = Field(..., description="Number of patients contacted")
    infections: list[str] = Field(
        ..., description="List of unique infections for this patient"
    )
    primary_infection: str = Field(..., description="Primary infection type")
    test_dates: list[str] = Field(..., description="List of test dates (YYYY-MM-DD)")
    contact_details: list[ContactDetailSchema] = Field(
        ..., description="Detailed contact information"
    )


class ContactEventSchema(BaseModel):
    """Schema for a contact event between two patients."""

    patient1: str = Field(..., description="First patient ID")
    patient2: str = Field(..., description="Second patient ID")
    location: str = Field(..., description="Location where contact occurred")
    organism: str = Field(..., description="Shared organism")
    contact_date: str = Field(..., description="Date of contact (YYYY-MM-DD)")
    days_from_test1: int = Field(..., description="Days from patient1's test date")
    days_from_test2: int = Field(..., description="Days from patient2's test date")


class ClusterContactSchema(BaseModel):
    """Schema for contact events within a cluster."""

    patient1: str = Field(..., description="First patient ID")
    patient2: str = Field(..., description="Second patient ID")
    location: str = Field(..., description="Location where contact occurred")
    contact_date: str = Field(..., description="Date of contact (YYYY-MM-DD)")


class ClusterPatientSchema(BaseModel):
    """Schema for patient data within a cluster."""

    patient_id: str = Field(..., description="Patient ID")
    infections: list[str] = Field(
        ..., description="List of infections for this patient"
    )
    test_dates: list[str] = Field(..., description="List of test dates (YYYY-MM-DD)")


class DateRangeSchema(BaseModel):
    """Schema for date range information."""

    earliest: str | None = Field(None, description="Earliest date (YYYY-MM-DD)")
    latest: str | None = Field(None, description="Latest date (YYYY-MM-DD)")


class ClusterSchema(BaseModel):
    """Schema for infection cluster data."""

    cluster_id: int = Field(..., description="Cluster identifier")
    patients: list[ClusterPatientSchema] = Field(
        ..., description="Patients in this cluster"
    )
    patient_count: int = Field(..., description="Number of patients in cluster")
    contact_count: int = Field(..., description="Number of contact events in cluster")
    infections: list[str] = Field(
        ..., description="List of unique infections in cluster"
    )
    infection_counts: dict[str, int] = Field(
        ..., description="Count of each infection type"
    )
    locations: list[str] = Field(..., description="Locations involved in cluster")
    date_range: DateRangeSchema = Field(
        ..., description="Date range of cluster activity"
    )
    contacts: list[ClusterContactSchema] = Field(
        ..., description="Contact events within cluster"
    )


class SummaryMetricsSchema(BaseModel):
    """Schema for summary metrics."""

    total_patients: int = Field(..., description="Total number of patients")
    connected_patients: int = Field(..., description="Number of patients with contacts")
    isolated_patients: int = Field(
        ..., description="Number of patients without contacts"
    )
    total_clusters: int = Field(..., description="Total number of infection clusters")
    total_contact_events: int = Field(..., description="Total number of contact events")
    infection_distribution: dict[str, int] = Field(
        ..., description="Distribution of infection types"
    )
    location_distribution: dict[str, int] = Field(
        ..., description="Distribution of contact locations"
    )
    largest_cluster_size: int = Field(..., description="Size of the largest cluster")


class MetadataSchema(BaseModel):
    """Schema for analysis metadata."""

    description: str = Field(..., description="Description of the analysis")
    generated_at: str = Field(..., description="Timestamp when analysis was generated")
    window_days: int = Field(
        ..., description="Contact window in days used for analysis"
    )
    data_format_version: str = Field(..., description="Data format version")


class InfectionDetectionResponse(BaseModel):
    """Complete response schema for infection detection analysis."""

    metadata: MetadataSchema = Field(..., description="Analysis metadata")
    summary: SummaryMetricsSchema = Field(..., description="Summary metrics")
    patients: dict[str, PatientInfoSchema] = Field(
        ..., description="Patient contact graph data"
    )
    clusters: list[ClusterSchema] = Field(..., description="Infection clusters")
    contacts: list[ContactEventSchema] = Field(..., description="All contact events")


class InfectionDetectionRequest(BaseModel):
    """Request schema for infection detection analysis."""

    window_days: int = Field(
        14, ge=1, le=365, description="Contact window in days (1-365)"
    )
    date_origin: str | None = Field(
        None, description="Origin date for analysis (YYYY-MM-DD)"
    )


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")


class TestCaseSchema(BaseModel):
    """Schema for patient test case information."""

    patient_id: str = Field(..., description="Patient ID")
    collection_date: str = Field(..., description="Test collection date (YYYY-MM-DD)")
    infection: str = Field(..., description="Infection/organism type")
    result: str = Field(..., description="Test result")


class TransferSchema(BaseModel):
    """Schema for patient transfer information."""

    patient_id: str = Field(..., description="Patient ID")
    location: str = Field(..., description="Ward/location name")
    ward_in_time: str = Field(..., description="Ward entry time (YYYY-MM-DD HH:MM:SS)")
    ward_out_time: str = Field(..., description="Ward exit time (YYYY-MM-DD HH:MM:SS)")
    duration_hours: float = Field(..., description="Duration of stay in hours")


class PatientDetailSchema(BaseModel):
    """Schema for detailed patient information."""

    patient_id: str = Field(..., description="Patient ID")
    test_cases: list[TestCaseSchema] = Field(
        ..., description="All test cases for this patient"
    )
    transfers: list[TransferSchema] = Field(
        ..., description="All transfers/stays for this patient"
    )
    positive_infections: list[str] = Field(
        ..., description="List of positive infections"
    )
    total_tests: int = Field(..., description="Total number of tests")
    total_transfers: int = Field(..., description="Total number of transfers/stays")
    first_positive_date: str | None = Field(
        None, description="Date of first positive test"
    )
    last_test_date: str | None = Field(None, description="Date of most recent test")


class PatientListResponse(BaseModel):
    """Response schema for patient list."""

    patients: list[PatientDetailSchema] = Field(
        ..., description="List of patient details"
    )
    total_patients: int = Field(..., description="Total number of patients")
    patients_with_positive_tests: int = Field(
        ..., description="Patients with at least one positive test"
    )
    total_positive_tests: int = Field(..., description="Total positive test count")
    unique_infections: list[str] = Field(
        ..., description="List of unique infection types found"
    )
