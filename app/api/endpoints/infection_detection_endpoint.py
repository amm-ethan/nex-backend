"""
API endpoints for infection detection functionality.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.infrastructure.schemas.response_schemas.infection_detection_schemas import (
    AllInfectionsVisualizationResponse,
    ClusterSummariesResponse,
    InfectionDetectionRequest,
    InfectionDetectionResponse,
    LocationRiskResponse,
    PatientDetailSchema,
    PatientListResponse,
    SpreadVisualizationResponse,
    SuperSpreadersResponse,
    TemporalPatternsResponse,
)
from app.infrastructure.services import (
    infection_detection_service,
    llm_analyzer_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/detect-clusters",
    response_model=InfectionDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect infection clusters",
    description="""
    Analyze hospital records to detect clusters of infections.
    
    The analysis includes:
    - Contact detection between patients
    - Cluster formation using transitive relationships
    - Detailed patient and cluster information
    - Summary statistics
    """,
)
async def detect_infection_clusters(
    request: InfectionDetectionRequest,
) -> InfectionDetectionResponse:
    """
    Detect infection clusters from hospital data.

    Args:
        request: Configuration for the analysis including window_days and date_origin

    Returns:
        InfectionDetectionResponse: Complete analysis results including clusters, contacts, and metrics

    Raises:
        HTTPException: If data files are missing or analysis fails
    """
    try:
        logger.info(
            f"Starting infection cluster detection with window_days={request.window_days}"
        )

        # Configure the service
        if request.date_origin:
            try:
                date_origin = datetime.strptime(request.date_origin, "%Y-%m-%d")
                infection_detection_service.date_origin = date_origin.date()
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid date_origin format. Use YYYY-MM-DD format.",
                )

        infection_detection_service.window_days = request.window_days

        # Run the detection pipeline
        results = await infection_detection_service.run_detection_pipeline()

        logger.info(
            f"Infection detection completed: {results['summary']['total_clusters']} clusters, "
            f"{results['summary']['total_contact_events']} contacts"
        )

        return InfectionDetectionResponse(**results)

    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data validation error: {str(e)}",
        )

    except Exception as e:
        logger.error(f"Unexpected error in infection detection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during infection cluster detection",
        )



@router.get(
    "/patients",
    response_model=PatientListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get all patients with their test cases and transfers",
    description="""
    Retrieve detailed information for all patients including:
    - All test cases (positive and negative)
    - All ward transfers with durations
    - Summary statistics per patient
    - Overall dataset statistics
    """,
)
async def get_all_patients() -> PatientListResponse:
    """
    Get detailed information for all patients.

    Returns:
        PatientListResponse: Complete patient data with tests and transfers

    Raises:
        HTTPException: If data files are missing or cannot be processed
    """
    try:
        logger.info("Fetching all patient details")

        patient_data = infection_detection_service.get_patient_details()

        logger.info(f"Retrieved data for {patient_data['total_patients']} patients")

        return PatientListResponse(**patient_data)

    except Exception as e:
        logger.error(f"Error fetching patient details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving patient data",
        )


@router.get(
    "/patients/{patient_id}",
    response_model=PatientDetailSchema,
    status_code=status.HTTP_200_OK,
    summary="Get specific patient details",
    description="""
    Retrieve detailed information for a specific patient including:
    - All test cases with dates and results
    - All ward transfers with entry/exit times and durations
    - Summary of positive infections
    - Test and transfer counts
    """,
)
async def get_patient_by_id(patient_id: str) -> PatientDetailSchema:
    """
    Get detailed information for a specific patient.

    Args:
        patient_id: The patient ID to retrieve details for

    Returns:
        PatientDetailSchema: Patient's test cases, transfers, and summary data

    Raises:
        HTTPException: If patient not found or data cannot be processed
    """
    try:
        logger.info(f"Fetching details for patient: {patient_id}")

        patient_data = infection_detection_service.get_patient_details(patient_id)

        if not patient_data["patients"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient {patient_id} not found",
            )

        patient_detail = patient_data["patients"][0]  # Single patient result

        logger.info(
            f"Retrieved details for patient {patient_id}: {patient_detail['total_tests']} tests, {patient_detail['total_transfers']} transfers"
        )

        return PatientDetailSchema(**patient_detail)

    except HTTPException:
        raise  # Re-raise HTTP exceptions

    except Exception as e:
        logger.error(f"Error fetching patient {patient_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving data for patient {patient_id}",
        )


@router.get(
    "/spread-visualization",
    response_model=AllInfectionsVisualizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get spread visualization data for all infections",
    description="""
    Generate comprehensive spread visualization data for React components including:
    - Timeline of all infection events (tests, contacts, transfers)
    - Network graph data (nodes and edges)
    - Spread events with confidence scores
    - Cross-infection analysis
    - Statistics for dashboard widgets
    
    Perfect for creating animated timeline visualizations, network graphs,
    and epidemiological analysis dashboards.
    """,
)
async def get_spread_visualization() -> AllInfectionsVisualizationResponse:
    """
    Get comprehensive spread visualization data for all infections.

    Returns:
        AllInfectionsVisualizationResponse: Complete visualization data including
        timeline events, network data, spread analysis, and statistics

    Raises:
        HTTPException: If data files are missing or analysis fails
    """
    try:
        logger.info("Generating spread visualization for all infections")

        visualization_data = infection_detection_service.generate_spread_visualization()

        logger.info(
            f"Generated spread visualization: {visualization_data['global_stats']['total_infections']} infections, "
            f"{visualization_data['global_stats']['total_patients']} patients"
        )

        return AllInfectionsVisualizationResponse(**visualization_data)

    except Exception as e:
        logger.error(f"Error generating spread visualization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating spread visualization data",
        )


@router.get(
    "/spread-visualization/{infection_type}",
    response_model=SpreadVisualizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get spread visualization data for specific infection",
    description="""
    Generate detailed spread visualization data for a specific infection type.
    
    Includes:
    - Chronological timeline of events (positive tests, contacts)
    - Potential transmission events with confidence scores
    - Network graph nodes and edges
    - Statistics and date ranges
    
    Use this for focused analysis of a single infection type's spread pattern.
    Common infection types: CRE, MRSA, VRE, ESBL
    """,
)
async def get_infection_spread_visualization(
    infection_type: str,
) -> SpreadVisualizationResponse:
    """
    Get spread visualization data for a specific infection type.

    Args:
        infection_type: The infection type to visualize (e.g., 'CRE', 'MRSA', 'VRE', 'ESBL')

    Returns:
        SpreadVisualizationResponse: Visualization data for the specified infection

    Raises:
        HTTPException: If infection type not found or data cannot be processed
    """
    try:
        logger.info(f"Generating spread visualization for infection: {infection_type}")

        visualization_data = infection_detection_service.generate_spread_visualization(
            infection_type
        )

        if not visualization_data.get("network_nodes"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No spread data found for infection type: {infection_type}",
            )

        logger.info(
            f"Generated {infection_type} visualization: {visualization_data['stats']['total_patients']} patients, "
            f"{visualization_data['stats']['total_contacts']} contacts"
        )

        return SpreadVisualizationResponse(**visualization_data)

    except HTTPException:
        raise  # Re-raise HTTP exceptions

    except Exception as e:
        logger.error(
            f"Error generating {infection_type} spread visualization: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating spread visualization for {infection_type}",
        )


@router.get(
    "/analysis/super-spreaders",
    response_model=SuperSpreadersResponse,
    status_code=status.HTTP_200_OK,
    summary="Get super spreader detection analysis",
    description="""
    Identify patients who likely caused multiple transmissions:
    
    - Risk scores based on transmission count, locations, and confidence
    - Transmission confidence averaging and infectious period analysis
    - Perfect for targeted intervention strategies
    - Identifies high-risk patients requiring special attention
    """,
)
async def get_super_spreaders() -> SuperSpreadersResponse:
    """
    Get super spreader detection analysis.

    Returns:
        SuperSpreadersResponse: Super spreader analysis with risk scores and metadata

    Raises:
        HTTPException: If data files are missing or analysis fails
    """
    try:
        logger.info("Generating super spreader analysis")

        super_spreaders_data = infection_detection_service.get_super_spreaders()

        logger.info(
            f"Super spreaders analysis generated: {len(super_spreaders_data['super_spreaders'])} super spreaders identified"
        )

        return SuperSpreadersResponse(**super_spreaders_data)

    except Exception as e:
        logger.error(f"Error generating super spreaders analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating super spreaders data",
        )


@router.get(
    "/analysis/location-risk",
    response_model=LocationRiskResponse,
    status_code=status.HTTP_200_OK,
    summary="Get location risk heatmap analysis",
    description="""
    Generate location-based infection risk analysis:
    
    - Infection rates, transmission events, and risk levels by location
    - Average stay duration and patient volume analysis
    - Actionable recommendations for high-risk areas
    - Risk categorization (LOW, MEDIUM, HIGH, CRITICAL)
    """,
)
async def get_location_risk_heatmaps() -> LocationRiskResponse:
    """
    Get location risk heatmap analysis.

    Returns:
        LocationRiskResponse: Location risk analysis with recommendations and metadata

    Raises:
        HTTPException: If data files are missing or analysis fails
    """
    try:
        logger.info("Generating location risk heatmap analysis")

        location_risk_data = infection_detection_service.get_location_risk_heatmaps()

        logger.info(
            f"Location risk analysis generated: {len(location_risk_data['location_risks'])} locations analyzed"
        )

        return LocationRiskResponse(**location_risk_data)

    except Exception as e:
        logger.error(f"Error generating location risk analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating location risk data",
        )


@router.get(
    "/llm",
    response_model=ClusterSummariesResponse,
    status_code=status.HTTP_200_OK,
    summary="Get LLM-generated clinical summaries for infection clusters",
    description="""
    Generate clinical summaries for all infection clusters using LLM analysis:
    
    - Uses LangChain + Ollama (preferred) or OpenAI as fallback
    - Plain-language summaries ≤120 words for clinicians
    - Risk assessment and actionable recommendations
    - Falls back to mock summaries if no LLM is available
    - Includes epidemiological insights and intervention guidance
    """,
)
async def get_cluster_summaries() -> ClusterSummariesResponse:
    """
    Generate clinical summaries for all infection clusters using LLM.

    Returns:
        ClusterSummariesResponse: Clinical summaries with risk assessment and recommendations

    Raises:
        HTTPException: If data files are missing or analysis fails
    """
    try:
        logger.info("Generating LLM-based cluster summaries")

        # First, ensure we have cluster data
        if not infection_detection_service.contacts:
            logger.info("No analysis found, running detection pipeline")
            await infection_detection_service.run_detection_pipeline()

        # Get cluster data
        clusters_data = infection_detection_service.generate_cluster_data()

        if not clusters_data:
            logger.warning("No clusters available for summary generation")
            return ClusterSummariesResponse(
                summaries=[],
                total_clusters=0,
                llm_status={
                    "ollama_available": llm_analyzer_service.ollama_available,
                    "openai_available": llm_analyzer_service.openai_available,
                },
                analysis_metadata={
                    "generated_at": datetime.now().isoformat(),
                    "message": "No infection clusters detected",
                },
            )

        # Generate summaries using LLM
        summaries = await llm_analyzer_service.generate_multiple_cluster_summaries(
            clusters_data
        )

        logger.info(f"Generated {len(summaries)} cluster summaries")

        # Convert to response format
        summary_schemas = []
        for summary in summaries:
            summary_schemas.append(
                {
                    "cluster_id": summary.cluster_id,
                    "clinical_summary": summary.clinical_summary,
                    "risk_level": summary.risk_level,
                    "key_insights": summary.key_insights,
                    "recommendations": summary.recommendations,
                    "generated_by": summary.generated_by,
                }
            )

        return ClusterSummariesResponse(
            summaries=summary_schemas,
            total_clusters=len(summaries),
            llm_status={
                "ollama_available": llm_analyzer_service.ollama_available,
                "openai_available": llm_analyzer_service.openai_available,
            },
            analysis_metadata={
                "generated_at": datetime.now().isoformat(),
                "successful_generations": len(
                    [s for s in summaries if s.generated_by != "error"]
                ),
                "llm_sources": list(set(s.generated_by for s in summaries)),
            },
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Required data files not found in app/data/",
        )

    except Exception as e:
        logger.error(f"Error generating cluster summaries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating cluster summaries",
        )


@router.get(
    "/analysis/temporal-patterns",
    response_model=TemporalPatternsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get temporal pattern analysis",
    description="""
    Analyze temporal patterns in infection transmission:
    
    - Peak transmission hours and high-risk days
    - Incubation periods and transmission velocity
    - Outbreak period identification with severity assessment
    - Seasonal trends and transmission dynamics
    """,
)
async def get_temporal_patterns() -> TemporalPatternsResponse:
    """
    Get temporal pattern analysis.

    Returns:
        TemporalPatternsResponse: Temporal pattern analysis with outbreak periods and metadata

    Raises:
        HTTPException: If data files are missing or analysis fails
    """
    try:
        logger.info("Generating temporal pattern analysis")

        temporal_patterns_data = infection_detection_service.get_temporal_patterns()

        logger.info(
            f"Temporal patterns analysis generated: {len(temporal_patterns_data['temporal_patterns'])} infection patterns"
        )

        return TemporalPatternsResponse(**temporal_patterns_data)

    except Exception as e:
        logger.error(f"Error generating temporal patterns analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating temporal patterns data",
        )


