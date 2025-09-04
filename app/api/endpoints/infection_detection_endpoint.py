"""
API endpoints for infection detection functionality.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.infrastructure.schemas.response_schemas.infection_detection_schemas import (
    InfectionDetectionRequest,
    InfectionDetectionResponse,
    PatientDetailSchema,
    PatientListResponse,
    SummaryMetricsSchema,
)
from app.infrastructure.services import infection_detection_service

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

    except FileNotFoundError as e:
        logger.error(f"Data files not found: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Required data files (microbiology.csv, transfers.csv) not found in app/data/",
        )

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
    "/summary",
    response_model=SummaryMetricsSchema,
    status_code=status.HTTP_200_OK,
    summary="Get infection detection summary",
    description="Get summary metrics from the last infection detection analysis.",
)
async def get_infection_summary() -> SummaryMetricsSchema:
    """
    Get summary metrics from infection detection.

    Returns:
        SummaryMetricsSchema: Summary statistics including patient counts, cluster counts, and distributions

    Raises:
        HTTPException: If no analysis has been run or data is unavailable
    """
    try:
        # Check if we have analyzed data
        if not infection_detection_service.contacts:
            # Run a quick analysis with default parameters
            logger.info(
                "No previous analysis found, running detection with default parameters"
            )
            await infection_detection_service.run_detection_pipeline()

        # Generate summary data
        graph_data = infection_detection_service.generate_graph_data()
        clusters = infection_detection_service.generate_cluster_data()
        summary = infection_detection_service.generate_summary_metrics(
            graph_data, clusters
        )

        return SummaryMetricsSchema(**summary)

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Required data files not found in app/data/",
        )

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating summary metrics",
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

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Required data files not found in app/data/",
        )

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

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Required data files not found in app/data/",
        )

    except Exception as e:
        logger.error(f"Error fetching patient {patient_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving data for patient {patient_id}",
        )


@router.get(
    "/data-status",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Check data file status",
    description="Check the availability and basic statistics of required data files.",
)
async def check_data_status() -> dict[str, Any]:
    """
    Check the status of data files required for infection detection.

    Returns:
        Dict containing data file status and basic statistics
    """
    try:
        data_dir = infection_detection_service.data_dir

        status_info = {"data_directory": str(data_dir), "files": {}}

        # Check microbiology file
        micro_file = infection_detection_service.microbiology_file
        if micro_file.exists():
            try:
                import pandas as pd

                df_micro = pd.read_csv(micro_file)
                status_info["files"]["microbiology.csv"] = {
                    "exists": True,
                    "size_bytes": micro_file.stat().st_size,
                    "row_count": len(df_micro),
                    "columns": list(df_micro.columns),
                }
            except Exception as e:
                status_info["files"]["microbiology.csv"] = {
                    "exists": True,
                    "error": f"Could not read file: {str(e)}",
                }
        else:
            status_info["files"]["microbiology.csv"] = {"exists": False}

        # Check transfers file
        transfers_file = infection_detection_service.transfers_file
        if transfers_file.exists():
            try:
                import pandas as pd

                df_transfers = pd.read_csv(transfers_file)
                status_info["files"]["transfers.csv"] = {
                    "exists": True,
                    "size_bytes": transfers_file.stat().st_size,
                    "row_count": len(df_transfers),
                    "columns": list(df_transfers.columns),
                }
            except Exception as e:
                status_info["files"]["transfers.csv"] = {
                    "exists": True,
                    "error": f"Could not read file: {str(e)}",
                }
        else:
            status_info["files"]["transfers.csv"] = {"exists": False}

        return status_info

    except Exception as e:
        logger.error(f"Error checking data status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking data status: {str(e)}",
        )
