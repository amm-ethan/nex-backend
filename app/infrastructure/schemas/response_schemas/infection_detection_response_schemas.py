"""
Response schemas for infection detection API endpoints.
"""

from typing import Any

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


class SpreadEventSchema(BaseModel):
    """Schema for infection spread event."""

    event_id: str = Field(..., description="Unique event identifier")
    source_patient: str = Field(..., description="Source patient ID")
    target_patient: str = Field(..., description="Target patient ID")
    infection: str = Field(..., description="Infection type")
    contact_date: str = Field(..., description="Date of contact (YYYY-MM-DD)")
    contact_location: str = Field(..., description="Location where contact occurred")
    source_test_date: str = Field(..., description="Source patient's test date")
    target_test_date: str = Field(..., description="Target patient's test date")
    days_between_tests: int = Field(..., description="Days between the two test dates")
    confidence_score: float = Field(
        ..., description="Confidence score (0-1) based on temporal proximity"
    )


class TimelineEventSchema(BaseModel):
    """Schema for timeline visualization events."""

    date: str = Field(..., description="Event date (YYYY-MM-DD)")
    event_type: str = Field(
        ..., description="Event type: 'positive_test', 'contact', 'transfer'"
    )
    patient_id: str = Field(..., description="Patient ID")
    infection: str | None = Field(None, description="Infection type (for test events)")
    location: str | None = Field(
        None, description="Location (for transfer/contact events)"
    )
    related_patient: str | None = Field(
        None, description="Related patient ID (for contact events)"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional event details"
    )


class NetworkNodeSchema(BaseModel):
    """Schema for network visualization node."""

    id: str = Field(..., description="Patient ID")
    infections: list[str] = Field(..., description="List of infections")
    primary_infection: str = Field(..., description="Most significant infection")
    first_positive_date: str | None = Field(
        None, description="First positive test date"
    )
    total_contacts: int = Field(..., description="Number of contacts")
    node_size: int = Field(..., description="Visual size based on contact count")
    cluster_id: int | None = Field(None, description="Cluster membership ID")


class NetworkEdgeSchema(BaseModel):
    """Schema for network visualization edge."""

    source: str = Field(..., description="Source patient ID")
    target: str = Field(..., description="Target patient ID")
    infection: str = Field(..., description="Shared infection")
    contact_date: str = Field(..., description="Contact date")
    location: str = Field(..., description="Contact location")
    strength: float = Field(..., description="Connection strength (0-1)")


class SpreadVisualizationResponse(BaseModel):
    """Response schema for spread visualization data."""

    infection_type: str = Field(..., description="Infection being visualized")
    timeline_events: list[TimelineEventSchema] = Field(
        ..., description="Chronological events"
    )
    spread_events: list[SpreadEventSchema] = Field(
        ..., description="Potential transmission events"
    )
    network_nodes: list[NetworkNodeSchema] = Field(
        ..., description="Patients as network nodes"
    )
    network_edges: list[NetworkEdgeSchema] = Field(
        ..., description="Connections between patients"
    )
    date_range: DateRangeSchema = Field(..., description="Date range of events")
    stats: dict[str, Any] = Field(..., description="Visualization statistics")


class AllInfectionsVisualizationResponse(BaseModel):
    """Response schema for all infections visualization."""

    infections: dict[str, SpreadVisualizationResponse] = Field(
        ..., description="Data by infection type"
    )
    combined_timeline: list[TimelineEventSchema] = Field(
        ..., description="Combined timeline across all infections"
    )
    cross_infection_events: list[dict[str, Any]] = Field(
        ..., description="Events involving multiple infections"
    )
    global_stats: dict[str, Any] = Field(..., description="Overall statistics")


class SuperSpreaderSchema(BaseModel):
    """Schema for super spreader analysis."""

    patient_id: str = Field(..., description="Patient ID")
    outbound_transmissions: int = Field(
        ..., description="Number of likely transmissions caused"
    )
    transmission_confidence_avg: float = Field(
        ..., description="Average confidence of transmissions"
    )
    locations_infected: list[str] = Field(
        ..., description="Locations where patient spread infections"
    )
    infection_period_days: int = Field(..., description="Duration of infectious period")
    infections: list[str] = Field(..., description="Types of infections spread")
    risk_score: float = Field(
        ..., description="Overall super spreader risk score (0-1)"
    )
    first_positive_date: str | None = Field(
        None, description="Date of first positive test"
    )
    total_contacts: int = Field(..., description="Total number of patient contacts")


class LocationRiskSchema(BaseModel):
    """Schema for location risk analysis."""

    location: str = Field(..., description="Location name")
    infection_rate: float = Field(
        ..., description="Proportion of patients who get infected"
    )
    avg_stay_duration: float = Field(..., description="Average stay duration in days")
    transmission_events: int = Field(..., description="Number of transmission events")
    total_patients: int = Field(
        ..., description="Total patients who stayed in location"
    )
    infected_patients: int = Field(
        ..., description="Number of patients who got infected"
    )
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    dominant_infections: list[str] = Field(
        ..., description="Most common infections in location"
    )
    recommended_actions: list[str] = Field(..., description="Suggested interventions")


class SuperSpreadersResponse(BaseModel):
    """Response schema for super spreader detection."""

    super_spreaders: list[SuperSpreaderSchema] = Field(
        ..., description="Super spreader analysis"
    )
    analysis_metadata: dict[str, Any] = Field(..., description="Analysis metadata")


class LocationRiskResponse(BaseModel):
    """Response schema for location risk analysis."""

    location_risks: list[LocationRiskSchema] = Field(
        ..., description="Location risk heatmap data"
    )
    analysis_metadata: dict[str, Any] = Field(..., description="Analysis metadata")


class ClusterSummarySchema(BaseModel):
    """Schema for LLM-generated cluster summary."""

    cluster_id: int = Field(..., description="Cluster identifier")
    clinical_summary: str = Field(..., description="Clinical summary (≤120 words)")
    risk_level: str = Field(
        ..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL, UNKNOWN"
    )
    key_insights: list[str] = Field(..., description="Key epidemiological insights")
    recommendations: list[str] = Field(..., description="Clinical recommendations")
    generated_by: str = Field(
        ..., description="LLM source: ollama, openai, mock, or error"
    )


class ClusterSummariesResponse(BaseModel):
    """Response schema for multiple cluster summaries."""

    summaries: list[ClusterSummarySchema] = Field(..., description="Cluster summaries")
    total_clusters: int = Field(..., description="Total number of clusters analyzed")
    llm_status: dict[str, Any] = Field(..., description="LLM availability status")
    analysis_metadata: dict[str, Any] = Field(..., description="Analysis metadata")
