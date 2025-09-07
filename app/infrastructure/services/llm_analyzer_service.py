"""
LLM Analyzer Service for generating clinical summaries of infection clusters.
Supports LangChain + Ollama with OpenAI fallback.
"""

import logging
from typing import Any

import requests
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class ClusterSummary(BaseModel):
    """Schema for cluster summary."""

    cluster_id: int
    clinical_summary: str
    risk_level: str
    key_insights: list[str]
    recommendations: list[str]
    generated_by: str  # "ollama", "openai", or "mock"


class LLMAnalyzerService:
    """Service for generating clinical summaries using LLM."""

    def __init__(self) -> None:
        self.ollama_available = False
        self.openai_available = False
        self._setup_llm_clients()

    def _setup_llm_clients(self) -> None:
        """Initialize LLM clients based on available configurations."""
        try:
            # Check for Ollama availability

            response = requests.get(f"{settings.OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                self.ollama_available = True
                logger.info("Ollama server detected and available")
        except Exception as e:
            logger.info(f"Ollama not available: {e}")

        # Check for OpenAI API key
        if settings.OPENAI_API_KEY:
            self.openai_available = True
            logger.info("OpenAI API key found")
        else:
            logger.info("OpenAI API key not found")

    def _get_cluster_prompt(self, cluster_data: dict[str, Any]) -> str:
        """Generate prompt for cluster analysis."""

        patients = cluster_data.get("patients", [])
        infections = cluster_data.get("infections", [])
        locations = cluster_data.get("locations", [])
        date_range = cluster_data.get("date_range", {})
        contacts = cluster_data.get("contacts", [])

        prompt = f"""
You are a hospital infection control specialist analyzing an infection cluster. 
Provide a concise clinical summary in exactly 120 words or fewer for healthcare professionals.

CLUSTER DATA:
- Cluster ID: {cluster_data.get("cluster_id")}
- Patients affected: {len(patients)}
- Infection types: {", ".join(infections)}
- Locations involved: {", ".join(locations)}
- Date range: {date_range.get("earliest")} to {date_range.get("latest")}
- Total contact events: {len(contacts)}

PATIENT DETAILS:
{self._format_patient_details(patients)}

CONTACT EVENTS:
{self._format_contact_events(contacts)}

REQUIREMENTS:
- Write in clinical, professional tone
- Focus on epidemiological significance
- Mention key locations and timeframe
- Include actionable insights
- Maximum 120 words
- Be factual and concise
"""
        return prompt.strip()

    def _format_patient_details(self, patients: list[dict]) -> str:
        """Format patient details for prompt."""
        details = []
        for patient in patients:
            patient_id = patient.get("patient_id", "Unknown")
            infections = patient.get("infections", [])
            test_dates = patient.get("test_dates", [])
            details.append(
                f"- {patient_id}: {', '.join(infections)} (tests: {', '.join(test_dates)})"
            )
        return "\n".join(details)

    def _format_contact_events(self, contacts: list[dict]) -> str:
        """Format contact events for prompt."""
        events = []
        for contact in contacts:
            patient1 = contact.get("patient1", "Unknown")
            patient2 = contact.get("patient2", "Unknown")
            location = contact.get("location", "Unknown")
            date = contact.get("contact_date", "Unknown")
            events.append(f"- {patient1} ↔ {patient2} at {location} on {date}")

        return "\n".join(events) if events else "No contact events available"

    async def _generate_with_ollama(self, prompt: str) -> str:
        """Generate summary using Ollama."""
        try:
            # Use a medical-focused model if available, otherwise fall back to llama2
            llm = OllamaLLM(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_URL,
            )

            response = await llm.ainvoke(prompt)
            logger.info("Generated summary using Ollama")
            return response.strip()

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def _generate_with_openai(self, prompt: str) -> str:
        """Generate summary using OpenAI."""
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,  # Low temperature for consistent, factual output
                max_tokens=200,
            )

            messages = [HumanMessage(content=prompt)]
            response = await llm.ainvoke(messages)

            logger.info("Generated summary using OpenAI")
            return response.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def _generate_mock_summary(self, cluster_data: dict[str, Any]) -> str:
        """Generate mock summary when no LLM is available."""
        cluster_id = cluster_data.get("cluster_id", 0)
        patient_count = len(cluster_data.get("patients", []))
        infections = cluster_data.get("infections", [])
        locations = cluster_data.get("locations", [])
        date_range = cluster_data.get("date_range", {})

        infection_text = ", ".join(infections) if infections else "multiple organisms"
        location_text = ", ".join(locations) if locations else "various locations"

        if len(locations) > 3:
            location_text += f" and {len(locations) - 3} other locations"

        summary = f"""
        Cluster {cluster_id} involves {patient_count} patients with {infection_text} infections 
        across {location_text}. Timeline spans from {date_range.get("earliest", "unknown")} 
        to {date_range.get("latest", "unknown")}. Epidemiological investigation suggests 
        potential healthcare-associated transmission. Recommend enhanced infection control 
        measures including contact precautions, environmental cleaning, and staff education. 
        Monitor for additional cases and consider molecular typing for outbreak confirmation. 
        Review antibiotic stewardship and isolation protocols in affected areas.
        """.strip()

        # Clean up spacing and ensure under 120 words
        words = summary.replace("\n", " ").split()
        if len(words) > 120:
            summary = " ".join(words[:120]) + "..."
        else:
            summary = " ".join(words)

        return summary

    def _extract_insights_and_recommendations(
        self, summary: str, cluster_data: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Extract key insights and recommendations from cluster data."""
        insights = []
        recommendations = []

        # Generate insights based on cluster characteristics
        patient_count = len(cluster_data.get("patients", []))
        infections = cluster_data.get("infections", [])
        locations = cluster_data.get("locations", [])

        if patient_count >= 5:
            insights.append("Large cluster indicating potential outbreak")

        if len(infections) > 1:
            insights.append(
                "Multi-organism cluster suggests environmental contamination"
            )

        if len(locations) > 3:
            insights.append("Wide geographic spread across multiple units")

        # Generate recommendations
        recommendations.append("Implement enhanced infection control measures")
        recommendations.append("Review and reinforce hand hygiene protocols")

        if len(locations) > 1:
            recommendations.append(
                "Coordinate infection control across all affected units"
            )

        if "ICU" in locations or "MICU" in locations:
            recommendations.append("Prioritize critical care infection prevention")

        return insights[:3], recommendations[:4]  # Limit to keep response concise

    def _determine_risk_level(self, cluster_data: dict[str, Any]) -> str:
        """Determine risk level based on cluster characteristics."""
        patient_count = len(cluster_data.get("patients", []))
        location_count = len(cluster_data.get("locations", []))

        # Simple risk scoring
        risk_score = 0

        if patient_count >= 10:
            risk_score += 3
        elif patient_count >= 5:
            risk_score += 2
        elif patient_count >= 3:
            risk_score += 1

        if location_count >= 4:
            risk_score += 2
        elif location_count >= 2:
            risk_score += 1

        # Check for high-risk locations
        high_risk_locations = ["ICU", "MICU", "NICU", "OR"]
        locations = cluster_data.get("locations", [])
        if any(loc in locations for loc in high_risk_locations):
            risk_score += 1

        if risk_score >= 5:
            return "CRITICAL"
        elif risk_score >= 3:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    async def generate_cluster_summary(
        self, cluster_data: dict[str, Any]
    ) -> ClusterSummary:
        """
        Generate clinical summary for an infection cluster.

        Args:
            cluster_data: Cluster information including patients, infections, locations, etc.

        Returns:
            ClusterSummary: Clinical summary with insights and recommendations
        """
        cluster_id = cluster_data.get("cluster_id", 0)
        prompt = self._get_cluster_prompt(cluster_data)

        # Try LLM generation in order of preference
        summary_text = ""
        generated_by = "mock"

        if self.ollama_available:
            try:
                summary_text = await self._generate_with_ollama(prompt)
                generated_by = "ollama"
            except Exception as e:
                logger.warning(f"Ollama failed, trying OpenAI: {e}")

        if not summary_text and self.openai_available:
            try:
                summary_text = await self._generate_with_openai(prompt)
                generated_by = "openai"
            except Exception as e:
                logger.warning(f"OpenAI failed, using mock: {e}")

        if not summary_text:
            summary_text = self._generate_mock_summary(cluster_data)
            generated_by = "mock"
            logger.info("Using mock summary - no LLM available")

        # Extract additional insights and recommendations
        insights, recommendations = self._extract_insights_and_recommendations(
            summary_text, cluster_data
        )
        risk_level = self._determine_risk_level(cluster_data)

        return ClusterSummary(
            cluster_id=cluster_id,
            clinical_summary=summary_text,
            risk_level=risk_level,
            key_insights=insights,
            recommendations=recommendations,
            generated_by=generated_by,
        )

    async def generate_multiple_cluster_summaries(
        self, clusters_data: list[dict[str, Any]]
    ) -> list[ClusterSummary]:
        """
        Generate summaries for multiple clusters.

        Args:
            clusters_data: List of cluster data dictionaries

        Returns:
            List of ClusterSummary objects
        """
        summaries = []

        for cluster_data in clusters_data:
            try:
                summary = await self.generate_cluster_summary(cluster_data)
                summaries.append(summary)
            except Exception as e:
                cluster_id = cluster_data.get("cluster_id", 0)
                logger.error(
                    f"Failed to generate summary for cluster {cluster_id}: {e}"
                )

                # Create fallback summary
                fallback_summary = ClusterSummary(
                    cluster_id=cluster_id,
                    clinical_summary="Unable to generate detailed summary. Manual review recommended.",
                    risk_level="UNKNOWN",
                    key_insights=["Summary generation failed"],
                    recommendations=["Manual clinical review required"],
                    generated_by="error",
                )
                summaries.append(fallback_summary)

        return summaries


# Global service instance
llm_analyzer_service = LLMAnalyzerService()
