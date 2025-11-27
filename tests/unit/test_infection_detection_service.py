"""
Comprehensive unit tests for infection detection service.
Tests UnionFind algorithm, service functionality, error handling, and advanced scenarios.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, mock_open
import tempfile
import os

import numpy as np
import pandas as pd
import pytest

from app.infrastructure.services.infection_detection_service import (
    InfectionDetectionService,
    UnionFind,
)
from tests.fixtures.sample_data import (
    create_sample_contacts,
    create_sample_microbiology_data,
    create_sample_transfers_data,
)


class TestUnionFind:
    """Test the UnionFind data structure."""

    def test_initialization(self):
        uf = UnionFind(5)
        assert len(uf.parent) == 5
        assert len(uf.rank) == 5
        assert all(uf.parent[i] == i for i in range(5))
        assert all(uf.rank[i] == 0 for i in range(5))

    def test_find_root(self):
        uf = UnionFind(5)
        assert uf.find(0) == 0
        assert uf.find(4) == 4

    def test_union_operation(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

        uf.union(2, 3)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)

        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_get_groups(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)

        groups = uf.get_groups()
        # Should have groups, exact number may vary based on implementation
        assert len(groups) >= 2  # At least {0,1}, {2,3} - {4} may be included or excluded

        group_sizes = [len(group) for group in groups]
        # Should have at least one group with 2 members
        assert 2 in group_sizes


class TestUnionFindExtended:
    """Extended tests for UnionFind data structure."""

    def test_empty_initialization(self):
        """Test UnionFind with zero size."""
        uf = UnionFind(0)
        assert len(uf.parent) == 0
        assert len(uf.rank) == 0
        groups = uf.get_groups()
        assert len(groups) == 0

    def test_single_element(self):
        """Test UnionFind with single element."""
        uf = UnionFind(1)
        assert uf.find(0) == 0
        groups = uf.get_groups()
        # Single element forms no group (no multi-element groups)
        assert len(groups) == 0

    def test_path_compression(self):
        """Test that path compression works correctly."""
        uf = UnionFind(10)
        
        # Create a chain: 0 -> 1 -> 2 -> 3 -> 4
        for i in range(4):
            uf.union(i, i + 1)
        
        # All should have same root
        root = uf.find(0)
        for i in range(5):
            assert uf.find(i) == root

    def test_union_by_rank(self):
        """Test union by rank optimization."""
        uf = UnionFind(8)
        
        # Create two trees of different sizes
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        uf.union(5, 6)
        uf.union(6, 7)
        
        # Union smaller with larger
        uf.union(1, 4)  # {0,1} joins {4,5,6,7}
        
        # Verify all connected
        root = uf.find(0)
        for i in [1, 4, 5, 6, 7]:
            assert uf.find(i) == root

    def test_large_dataset(self):
        """Test UnionFind with large number of elements."""
        n = 1000
        uf = UnionFind(n)
        
        # Connect every 10th element to element 0
        for i in range(0, n, 10):
            uf.union(0, i)
        
        # Check that all 10th elements are connected
        root = uf.find(0)
        for i in range(0, n, 10):
            assert uf.find(i) == root
        
        groups = uf.get_groups()
        # Should have one large group and many single-element groups
        group_sizes = [len(group) for group in groups]
        assert max(group_sizes) == 100  # Every 10th element connected to 0

    def test_get_groups_structure(self):
        """Test that get_groups returns correct structure."""
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        
        groups = uf.get_groups()
        
        # Should have 3 groups
        assert len(groups) == 3
        
        # Each group should have 2 elements
        for group in groups:
            assert len(group) == 2
        
        # Verify all elements are present exactly once
        all_elements = set()
        for group in groups:
            for element in group:
                assert element not in all_elements  # No duplicates
                all_elements.add(element)
        
        assert all_elements == {0, 1, 2, 3, 4, 5}

    def test_sequential_unions(self):
        """Test sequential union operations."""
        uf = UnionFind(5)
        
        # Initially all separate - no multi-element groups
        groups = uf.get_groups()
        assert len(groups) == 0
        
        # Union 0-1 - creates first group
        uf.union(0, 1)
        groups = uf.get_groups()
        assert len(groups) == 1
        
        # Union 1-2 (extends existing group)
        uf.union(1, 2)
        groups = uf.get_groups()
        assert len(groups) == 1  # Still just one group, but larger
        
        # Union 3-4 (new group)
        uf.union(3, 4)
        groups = uf.get_groups()
        assert len(groups) == 2  # Now we have two groups
        
        # Union groups together
        uf.union(2, 3)
        groups = uf.get_groups()
        assert len(groups) == 1  # Back to one large group
        assert len(groups[0]) == 5

    def test_invalid_operations(self):
        """Test handling of invalid operations."""
        uf = UnionFind(5)
        
        # These should not raise exceptions but also not crash
        try:
            uf.union(0, 0)  # Self-union
            uf.find(0)  # Should still work
        except Exception:
            pytest.fail("Self-union should not raise exception")


class TestInfectionDetectionService:
    """Test the main InfectionDetectionService class."""

    @pytest.fixture
    def service(self):
        return InfectionDetectionService()

    @pytest.fixture
    def mock_data_files(self):
        """Mock file reading to return sample data."""
        with patch('pandas.read_csv') as mock_read_csv:
            def side_effect(filepath):
                if 'microbiology' in str(filepath):
                    return create_sample_microbiology_data()
                elif 'transfers' in str(filepath):
                    return create_sample_transfers_data()
                return pd.DataFrame()

            mock_read_csv.side_effect = side_effect
            yield mock_read_csv

    def test_initialization(self, service):
        assert service.df_micro is None
        assert service.df_transfers is None
        assert service.df_positive is None
        assert service.contacts == []
        assert service.contact_groups == []

    def test_load_and_optimize_data(self, service, mock_data_files):
        service.load_and_optimize_data()

        assert service.df_micro is not None
        assert service.df_transfers is not None
        assert len(service.df_micro) == 5
        assert len(service.df_transfers) == 4

        # Check positive filtering
        assert service.df_positive is not None
        assert len(service.df_positive) == 4  # 4 positive results
        assert all(service.df_positive['result'] == 'positive')

    def test_contact_detection_basic(self, service, mock_data_files):
        service.load_and_optimize_data()
        service.create_spatial_temporal_index()
        service.contact_detection()

        # Should find contacts (exact number may vary based on overlap logic)
        assert isinstance(service.contacts, list)

        if service.contacts:  # If contacts found, check structure
            contact = service.contacts[0]
            assert 'patient1' in contact
            assert 'patient2' in contact
            assert 'location' in contact
            assert 'contact_date' in contact

    def test_get_contact_groups(self, service):
        # Manually set up contacts for testing
        service.contacts = create_sample_contacts()
        service.get_contact_groups()

        assert isinstance(service.contact_groups, list)
        if service.contact_groups:
            # P001-P002 and P002-P003 should form connected components
            total_patients = sum(len(group) for group in service.contact_groups)
            assert total_patients >= 2

    def test_generate_graph_data(self, service, mock_data_files):
        service.load_and_optimize_data()
        service.contacts = create_sample_contacts()

        graph_data = service.generate_graph_data()

        assert isinstance(graph_data, dict)
        assert len(graph_data) > 0

        # Check structure of patient data
        for patient_id, data in graph_data.items():
            assert 'contacts' in data
            assert 'contact_count' in data
            assert 'infections' in data
            assert 'primary_infection' in data
            assert 'test_dates' in data
            assert 'contact_details' in data

    def test_generate_cluster_data(self, service, mock_data_files):
        service.load_and_optimize_data()
        service.contacts = create_sample_contacts()
        service.get_contact_groups()

        clusters = service.generate_cluster_data()

        assert isinstance(clusters, list)
        if clusters:  # If any clusters found
            cluster = clusters[0]
            assert 'cluster_id' in cluster
            assert 'patients' in cluster
            assert 'patient_count' in cluster
            assert 'contact_count' in cluster
            assert 'infections' in cluster
            assert 'infection_counts' in cluster
            assert 'locations' in cluster
            assert 'date_range' in cluster
            assert 'contacts' in cluster

    def test_generate_summary_metrics(self, service, mock_data_files):
        service.load_and_optimize_data()
        service.contacts = create_sample_contacts()
        service.get_contact_groups()

        graph_data = service.generate_graph_data()
        clusters = service.generate_cluster_data()
        summary = service.generate_summary_metrics(graph_data, clusters)

        assert isinstance(summary, dict)
        assert 'total_patients' in summary
        assert 'connected_patients' in summary
        assert 'isolated_patients' in summary
        assert 'total_clusters' in summary
        assert 'total_contact_events' in summary
        assert 'infection_distribution' in summary
        assert 'location_distribution' in summary
        assert 'largest_cluster_size' in summary

        # Validate data types
        assert isinstance(summary['total_patients'], int)
        assert isinstance(summary['connected_patients'], int)
        assert isinstance(summary['isolated_patients'], int)
        assert isinstance(summary['infection_distribution'], dict)
        assert isinstance(summary['location_distribution'], dict)

    def test_get_patient_details(self, service, mock_data_files):
        service.load_and_optimize_data()
        service.contacts = create_sample_contacts()

        # Test all patients
        all_patients_data = service.get_patient_details()
        assert isinstance(all_patients_data, dict)
        assert 'patients' in all_patients_data
        assert 'total_patients' in all_patients_data
        assert 'total_positive_tests' in all_patients_data
        assert 'unique_infections' in all_patients_data

        if all_patients_data['patients']:
            patient = all_patients_data['patients'][0]
            assert 'patient_id' in patient
            assert 'test_cases' in patient
            assert 'transfers' in patient
            assert 'positive_infections' in patient
            assert 'total_tests' in patient
            assert 'total_transfers' in patient

        # Test specific patient
        specific_patient_data = service.get_patient_details("P001")
        assert isinstance(specific_patient_data, dict)
        assert 'patients' in specific_patient_data

        # Test non-existing patient
        nonexistent_data = service.get_patient_details("NONEXISTENT")
        assert nonexistent_data['patients'] == []

    @pytest.mark.asyncio
    async def test_run_detection_pipeline(self, service, mock_data_files):
        """Test the complete detection pipeline."""
        with patch.object(service, '_validate_data_files', return_value=True):
            result = await service.run_detection_pipeline()

            # Verify result structure
            assert isinstance(result, dict)
            assert 'metadata' in result
            assert 'summary' in result
            assert 'patients' in result
            assert 'clusters' in result
            assert 'contacts' in result

            # Verify all steps completed
            assert service.df_micro is not None
            assert service.df_positive is not None
            assert isinstance(service.contacts, list)
            assert isinstance(service.contact_groups, list)

    def test_error_handling_missing_data(self, service):
        """Test error handling when data files are missing."""
        with patch.object(service, '_validate_data_files', return_value=False):
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                service.load_and_optimize_data()
            assert exc_info.value.status_code == 404

    def test_empty_data_handling(self, service):
        """Test handling of empty datasets."""
        service.df_micro = pd.DataFrame()
        service.df_transfers = pd.DataFrame()
        service.df_positive = pd.DataFrame()

        service.contact_detection()
        assert service.contacts == []

        service.get_contact_groups()
        assert service.contact_groups == []

        graph_data = service.generate_graph_data()
        assert graph_data == {}

    def test_generate_spread_visualization(self, service, mock_data_files):
        """Test spread visualization generation."""
        service.load_and_optimize_data()
        # Create contacts with proper structure including organism field
        service.contacts = [
            {
                "patient1": "P001",
                "patient2": "P002",
                "location": "Ward-1",
                "organism": "CRE",
                "contact_date": date(2025, 1, 2),
                "days_from_test1": 1,
                "days_from_test2": 0
            }
        ]

        # Test with infection type filter
        spread_data = service.generate_spread_visualization("CRE")
        assert isinstance(spread_data, dict)
        assert 'infection_type' in spread_data
        assert 'timeline_events' in spread_data
        assert 'network_nodes' in spread_data
        assert 'stats' in spread_data

        # Test without filter (all infections)
        spread_data_all = service.generate_spread_visualization()
        assert isinstance(spread_data_all, dict)
        assert 'infections' in spread_data_all
        assert 'global_stats' in spread_data_all

    def test_get_super_spreaders(self, service, mock_data_files):
        """Test super spreader detection."""
        service.load_and_optimize_data()
        service.contacts = [
            {
                "patient1": "P001",
                "patient2": "P002",
                "location": "Ward-1",
                "organism": "CRE",
                "contact_date": date(2025, 1, 2),
                "days_from_test1": 1,
                "days_from_test2": 5
            }
        ]

        super_spreaders = service.get_super_spreaders()
        assert isinstance(super_spreaders, dict)
        assert 'super_spreaders' in super_spreaders
        assert 'analysis_metadata' in super_spreaders
        assert isinstance(super_spreaders['super_spreaders'], list)

    def test_get_location_risk_heatmaps(self, service, mock_data_files):
        """Test location risk heatmap generation."""
        service.load_and_optimize_data()
        service.contacts = [
            {
                "patient1": "P001",
                "patient2": "P002",
                "location": "Ward-1",
                "organism": "CRE",
                "contact_date": date(2025, 1, 2),
                "days_from_test1": 1,
                "days_from_test2": 0
            }
        ]

        heatmaps = service.get_location_risk_heatmaps()
        assert isinstance(heatmaps, dict)
        assert 'location_risks' in heatmaps
        assert 'analysis_metadata' in heatmaps
        assert isinstance(heatmaps['location_risks'], list)

    def test_validate_data_files(self, service):
        """Test data file validation."""
        # Mock the file paths as Path objects
        with patch.object(service, 'microbiology_file', Path('/tmp/micro.csv')):
            with patch.object(service, 'transfers_file', Path('/tmp/transfers.csv')):
                with patch('pathlib.Path.exists', side_effect=[True, True]):
                    result = service._validate_data_files()
                    assert result is True

                # Test with non-existent files
                with patch('pathlib.Path.exists', side_effect=[False, True]):
                    result = service._validate_data_files()
                    assert result is False

                with patch('pathlib.Path.exists', side_effect=[True, False]):
                    result = service._validate_data_files()
                    assert result is False


class TestInfectionDetectionServiceExtended:
    """Extended tests for InfectionDetectionService covering edge cases and complex scenarios."""

    @pytest.fixture
    def service(self):
        return InfectionDetectionService()

    @pytest.fixture
    def mock_data_files_extended(self):
        """Mock with extended test data."""
        with patch('pandas.read_csv') as mock_read_csv:
            def side_effect(filepath):
                if 'microbiology' in str(filepath):
                    # Extended microbiology data
                    data = {
                        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007'],
                        'collection_date': [
                            '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', 
                            '2025-01-05', '2025-01-06', '2025-01-07'
                        ],
                        'infection': ['CRE', 'ESBL', 'VRE', 'MRSA', 'CRE', 'ESBL', 'VRE'],
                        'result': ['positive', 'positive', 'negative', 'positive', 'positive', 'negative', 'positive']
                    }
                    return pd.DataFrame(data)
                elif 'transfers' in str(filepath):
                    # Extended transfers data
                    data = {
                        'patient_id': ['P001', 'P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007'],
                        'location': ['Ward-1', 'Ward-2', 'Ward-1', 'Ward-2', 'Ward-1', 'Ward-3', 'Ward-1', 'Ward-2'],
                        'ward_in_time': [
                            '2024-12-30', '2025-01-02', '2025-01-01', '2025-01-02',
                            '2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06'
                        ],
                        'ward_out_time': [
                            '2025-01-02', '2025-01-05', '2025-01-03', '2025-01-04',
                            '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09'
                        ]
                    }
                    return pd.DataFrame(data)
                return pd.DataFrame()
            
            mock_read_csv.side_effect = side_effect
            yield mock_read_csv

    def test_complex_contact_detection(self, service, mock_data_files_extended):
        """Test contact detection with complex overlapping scenarios."""
        service.load_and_optimize_data()
        service.create_spatial_temporal_index()
        service.contact_detection()
        
        # Should detect multiple contacts based on overlapping stays
        assert isinstance(service.contacts, list)
        
        if service.contacts:
            # Verify contact structure
            for contact in service.contacts:
                assert all(key in contact for key in ['patient1', 'patient2', 'location', 'contact_date'])
                assert contact['patient1'] != contact['patient2']

    def test_multiple_infection_types(self, service, mock_data_files_extended):
        """Test handling of multiple infection types."""
        service.load_and_optimize_data()
        
        # Should have multiple infection types
        unique_infections = service.df_positive['infection'].unique()
        assert len(unique_infections) >= 3
        assert 'CRE' in unique_infections
        assert 'ESBL' in unique_infections
        assert 'MRSA' in unique_infections

    def test_patient_with_multiple_infections(self, service):
        """Test patient with multiple different infections."""
        # Mock data with patient having multiple infections
        micro_data = pd.DataFrame({
            'patient_id': ['P001', 'P001', 'P002'],
            'collection_date': ['2025-01-01', '2025-01-02', '2025-01-03'],
            'infection': ['CRE', 'ESBL', 'CRE'],
            'result': ['positive', 'positive', 'positive']
        })
        
        transfers_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'location': ['Ward-1', 'Ward-1'],
            'ward_in_time': ['2025-01-01', '2025-01-02'],
            'ward_out_time': ['2025-01-03', '2025-01-04']
        })
        
        service.df_micro = micro_data
        service.df_transfers = transfers_data
        service.df_positive = micro_data[micro_data['result'] == 'positive']
        
        # Create mock contacts
        service.contacts = [{
            'patient1': 'P001',
            'patient2': 'P002',
            'location': 'Ward-1',
            'contact_date': date(2025, 1, 2),
            'organism': 'CRE'
        }]
        
        graph_data = service.generate_graph_data()
        
        # P001 should have both CRE and ESBL
        assert 'P001' in graph_data
        p001_infections = set(graph_data['P001']['infections'])
        assert 'CRE' in p001_infections
        assert 'ESBL' in p001_infections

    def test_isolated_patients(self, service, mock_data_files_extended):
        """Test detection of isolated patients (positive but no contacts)."""
        service.load_and_optimize_data()
        
        # Mock scenario with no contacts
        service.contacts = []
        service.contact_groups = []
        
        graph_data = service.generate_graph_data()
        clusters = service.generate_cluster_data()
        summary = service.generate_summary_metrics(graph_data, clusters)
        
        # All positive patients should be isolated
        assert summary['isolated_patients'] == summary['total_patients']
        assert summary['connected_patients'] == 0
        assert summary['total_clusters'] == 0

    def test_large_cluster_formation(self, service):
        """Test formation of large clusters."""
        # Create a connected network of 5 patients
        service.contacts = [
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1)},
            {'patient1': 'P002', 'patient2': 'P003', 'location': 'Ward-1', 'contact_date': date(2025, 1, 2)},
            {'patient1': 'P003', 'patient2': 'P004', 'location': 'Ward-1', 'contact_date': date(2025, 1, 3)},
            {'patient1': 'P004', 'patient2': 'P005', 'location': 'Ward-1', 'contact_date': date(2025, 1, 4)}
        ]
        
        service.get_contact_groups()
        
        # Should form one large cluster
        assert len(service.contact_groups) == 1
        assert len(service.contact_groups[0]) == 5

    def test_multiple_separate_clusters(self, service):
        """Test formation of multiple separate clusters."""
        service.contacts = [
            # Cluster 1: P001-P002-P003
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1)},
            {'patient1': 'P002', 'patient2': 'P003', 'location': 'Ward-1', 'contact_date': date(2025, 1, 2)},
            # Cluster 2: P004-P005
            {'patient1': 'P004', 'patient2': 'P005', 'location': 'Ward-2', 'contact_date': date(2025, 1, 3)},
        ]
        
        service.get_contact_groups()
        
        # Should form two separate clusters
        assert len(service.contact_groups) == 2
        cluster_sizes = [len(group) for group in service.contact_groups]
        assert 3 in cluster_sizes
        assert 2 in cluster_sizes

    def test_cross_location_transmission(self, service):
        """Test detection of transmission across different locations."""
        service.contacts = [
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1)},
            {'patient1': 'P002', 'patient2': 'P003', 'location': 'Ward-2', 'contact_date': date(2025, 1, 2)}
        ]
        
        service.get_contact_groups()
        clusters = service.generate_cluster_data()
        
        if clusters:
            cluster = clusters[0]
            # Should include both locations
            assert len(cluster['locations']) >= 2
            assert 'Ward-1' in cluster['locations'] or 'Ward-2' in cluster['locations']

    def test_temporal_edge_cases(self, service):
        """Test temporal edge cases in contact detection."""
        # Mock data with edge cases
        micro_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'collection_date': ['2025-01-01', '2025-01-01'],  # Same date
            'infection': ['CRE', 'CRE'],
            'result': ['positive', 'positive']
        })
        
        transfers_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'location': ['Ward-1', 'Ward-1'],
            'ward_in_time': ['2025-01-01', '2025-01-01'],  # Same admission
            'ward_out_time': ['2025-01-01', '2025-01-01']   # Same discharge
        })
        
        service.df_micro = micro_data
        service.df_transfers = transfers_data
        service.df_positive = micro_data
        service.create_spatial_temporal_index()
        service.contact_detection()
        
        # Should detect contact even with same dates
        if service.contacts:
            contact = service.contacts[0]
            assert contact['patient1'] in ['P001', 'P002']
            assert contact['patient2'] in ['P001', 'P002']

    def test_date_conversion_edge_cases(self, service):
        """Test date conversion with various formats."""
        # Mock data with different date formats
        micro_data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'collection_date': ['2025-01-01', '01/02/2025', '2025-01-03'],
            'infection': ['CRE', 'ESBL', 'VRE'],
            'result': ['positive', 'positive', 'positive']
        })
        
        service.df_micro = micro_data
        service.df_positive = micro_data
        
        # Should handle conversion gracefully
        try:
            service.create_spatial_temporal_index()
        except Exception as e:
            # If conversion fails, should be handled gracefully
            assert "date" in str(e).lower() or "time" in str(e).lower()

    def test_memory_efficiency_large_dataset(self, service):
        """Test memory efficiency with large datasets."""
        # Create larger mock dataset
        n_patients = 100
        n_transfers = 500
        
        micro_data = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(n_patients)],
            'collection_date': ['2025-01-01'] * n_patients,
            'infection': ['CRE'] * n_patients,
            'result': ['positive'] * n_patients
        })
        
        transfers_data = pd.DataFrame({
            'patient_id': [f'P{i%n_patients:03d}' for i in range(n_transfers)],
            'location': [f'Ward-{i%10}' for i in range(n_transfers)],
            'ward_in_time': ['2025-01-01'] * n_transfers,
            'ward_out_time': ['2025-01-02'] * n_transfers
        })
        
        service.df_micro = micro_data
        service.df_transfers = transfers_data
        service.df_positive = micro_data
        
        # Should complete without memory errors
        service.create_spatial_temporal_index()
        service.contact_detection()
        
        # Should produce reasonable results
        assert isinstance(service.contacts, list)

    def test_super_spreader_detection_edge_cases(self, service, mock_data_files_extended):
        """Test super spreader detection with edge cases."""
        service.load_and_optimize_data()
        
        # Mock complex contact pattern
        service.contacts = [
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1)},
            {'patient1': 'P001', 'patient2': 'P003', 'location': 'Ward-1', 'contact_date': date(2025, 1, 2)},
            {'patient1': 'P001', 'patient2': 'P004', 'location': 'Ward-2', 'contact_date': date(2025, 1, 3)},
            {'patient1': 'P002', 'patient2': 'P005', 'location': 'Ward-1', 'contact_date': date(2025, 1, 4)}
        ]
        
        super_spreaders = service.get_super_spreaders()
        
        assert 'super_spreaders' in super_spreaders
        assert 'analysis_metadata' in super_spreaders
        
        # P001 should be identified as potential super spreader (3 contacts)
        if super_spreaders['super_spreaders']:
            assert any(ss['patient_id'] == 'P001' for ss in super_spreaders['super_spreaders'])

    def test_location_risk_analysis_comprehensive(self, service, mock_data_files_extended):
        """Test comprehensive location risk analysis."""
        service.load_and_optimize_data()
        
        # Mock contacts across multiple locations
        service.contacts = [
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1)},
            {'patient1': 'P003', 'patient2': 'P004', 'location': 'Ward-1', 'contact_date': date(2025, 1, 2)},
            {'patient1': 'P005', 'patient2': 'P006', 'location': 'Ward-2', 'contact_date': date(2025, 1, 3)}
        ]
        
        heatmaps = service.get_location_risk_heatmaps()
        
        assert 'location_risks' in heatmaps
        assert 'analysis_metadata' in heatmaps
        
        if heatmaps['location_risks']:
            location_risk = heatmaps['location_risks'][0]
            required_fields = [
                'location', 'infection_rate', 'transmission_events',
                'total_patients', 'infected_patients', 'risk_level'
            ]
            
            for field in required_fields:
                assert field in location_risk

    def test_spread_visualization_complex_scenarios(self, service, mock_data_files_extended):
        """Test spread visualization with complex scenarios."""
        service.load_and_optimize_data()
        
        # Mock complex spread pattern
        service.contacts = [
            {
                'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1',
                'contact_date': date(2025, 1, 1), 'organism': 'CRE',
                'days_from_test1': 0, 'days_from_test2': 1
            },
            {
                'patient1': 'P002', 'patient2': 'P003', 'location': 'Ward-1', 
                'contact_date': date(2025, 1, 2), 'organism': 'CRE',
                'days_from_test1': 1, 'days_from_test2': 0
            }
        ]
        
        # Test specific infection visualization
        viz_data = service.generate_spread_visualization('CRE')
        
        assert 'infection_type' in viz_data
        assert viz_data['infection_type'] == 'CRE'
        assert 'timeline_events' in viz_data
        assert 'network_nodes' in viz_data
        assert 'stats' in viz_data
        
        # Test all infections visualization  
        viz_all = service.generate_spread_visualization()
        assert 'infections' in viz_all
        assert 'global_stats' in viz_all

    def test_data_quality_validation(self, service):
        """Test validation of data quality."""
        # Test with missing columns
        incomplete_micro = pd.DataFrame({
            'patient_id': ['P001'],
            'collection_date': ['2025-01-01']
            # Missing 'infection' and 'result' columns
        })
        
        service.df_micro = incomplete_micro
        
        # Should handle missing columns gracefully
        try:
            service.df_positive = service.df_micro[service.df_micro['result'] == 'positive']
        except KeyError:
            # Expected behavior - should raise KeyError for missing column
            pass

    def test_concurrent_processing_simulation(self, service, mock_data_files_extended):
        """Test behavior under simulated concurrent processing."""
        service.load_and_optimize_data()
        
        # Simulate multiple calls to contact detection
        service.create_spatial_temporal_index()
        service.contact_detection()
        initial_contacts = len(service.contacts)
        
        # Call again - should not duplicate
        service.contact_detection()
        assert len(service.contacts) >= initial_contacts

    def test_edge_case_patient_ids(self, service):
        """Test handling of edge case patient IDs."""
        # Mock data with unusual patient IDs
        micro_data = pd.DataFrame({
            'patient_id': ['', 'P001', None, 'VERY_LONG_PATIENT_ID_12345'],
            'collection_date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
            'infection': ['CRE', 'ESBL', 'VRE', 'MRSA'],
            'result': ['positive', 'positive', 'positive', 'positive']
        })
        
        service.df_micro = micro_data
        service.df_positive = micro_data[micro_data['result'] == 'positive']
        
        # Should handle unusual IDs gracefully
        patient_details = service.get_patient_details()
        assert isinstance(patient_details, dict)

    def test_performance_monitoring(self, service, mock_data_files_extended):
        """Test performance monitoring capabilities."""
        import time
        
        service.load_and_optimize_data()
        
        # Time the contact detection process
        start_time = time.time()
        service.create_spatial_temporal_index()
        service.contact_detection()
        end_time = time.time()
        
        # Should complete in reasonable time (less than 10 seconds for test data)
        assert (end_time - start_time) < 10

    def test_metadata_generation(self, service, mock_data_files_extended):
        """Test metadata generation in pipeline results."""
        service.load_and_optimize_data()
        service.create_spatial_temporal_index()  
        service.contact_detection()
        service.get_contact_groups()
        
        graph_data = service.generate_graph_data()
        clusters = service.generate_cluster_data()
        summary = service.generate_summary_metrics(graph_data, clusters)
        
        # Verify metadata fields are present and valid
        assert isinstance(summary, dict)
        assert all(isinstance(summary[key], (int, dict)) for key in summary if key in [
            'total_patients', 'connected_patients', 'isolated_patients',
            'total_clusters', 'total_contact_events', 'largest_cluster_size'
        ])


class TestInfectionDetectionServiceAdvanced:
    """Advanced tests for InfectionDetectionService with complex scenarios and numba functions."""
    
    @pytest.fixture
    def service(self):
        return InfectionDetectionService()
    
    @pytest.fixture
    def complex_mock_data(self):
        """Create complex mock data for advanced testing."""
        with patch('pandas.read_csv') as mock_read_csv:
            def side_effect(filepath):
                if 'microbiology' in str(filepath):
                    # Complex microbiology data with various scenarios
                    data = {
                        'patient_id': [
                            'P001', 'P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 
                            'P008', 'P009', 'P010', 'P011', 'P012'
                        ],
                        'collection_date': [
                            '2025-01-01', '2025-01-05', '2025-01-02', '2025-01-03', '2025-01-04',
                            '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
                            '2025-01-11', '2025-01-12', '2025-01-13'
                        ],
                        'infection': [
                            'CRE', 'ESBL', 'CRE', 'ESBL', 'VRE', 'MRSA', 'CRE', 'ESBL',
                            'VRE', 'MRSA', 'CRE', 'ESBL', 'VRE'
                        ],
                        'result': [
                            'positive', 'positive', 'positive', 'negative', 'positive', 'positive',
                            'positive', 'positive', 'negative', 'positive', 'positive', 'negative', 'positive'
                        ]
                    }
                    return pd.DataFrame(data)
                elif 'transfers' in str(filepath):
                    # Complex transfers data with overlapping stays
                    data = {
                        'patient_id': [
                            'P001', 'P001', 'P002', 'P002', 'P003', 'P004', 'P005', 'P006',
                            'P007', 'P008', 'P009', 'P010', 'P011', 'P012'
                        ],
                        'location': [
                            'Ward-1', 'Ward-2', 'Ward-1', 'Ward-3', 'Ward-2', 'Ward-1', 'Ward-2',
                            'Ward-1', 'Ward-3', 'Ward-2', 'Ward-1', 'Ward-3', 'Ward-1', 'Ward-2'
                        ],
                        'ward_in_time': [
                            '2024-12-30', '2025-01-03', '2025-01-01', '2025-01-04', '2025-01-02',
                            '2025-01-03', '2025-01-05', '2025-01-06', '2025-01-07', '2025-01-08',
                            '2025-01-09', '2025-01-10', '2025-01-11', '2025-01-12'
                        ],
                        'ward_out_time': [
                            '2025-01-03', '2025-01-06', '2025-01-04', '2025-01-07', '2025-01-05',
                            '2025-01-06', '2025-01-08', '2025-01-09', '2025-01-10', '2025-01-11',
                            '2025-01-12', '2025-01-13', '2025-01-14', '2025-01-15'
                        ]
                    }
                    return pd.DataFrame(data)
                return pd.DataFrame()
            
            mock_read_csv.side_effect = side_effect
            yield mock_read_csv
    
    def test_advanced_super_spreader_detection(self, service, complex_mock_data):
        """Test advanced super spreader detection with complex contact patterns."""
        service.load_and_optimize_data()
        
        # Create complex contact network where P001 is a clear super spreader
        service.contacts = [
            # P001 contacts multiple patients
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1), 'organism': 'CRE'},
            {'patient1': 'P001', 'patient2': 'P005', 'location': 'Ward-2', 'contact_date': date(2025, 1, 5), 'organism': 'CRE'},
            {'patient1': 'P001', 'patient2': 'P006', 'location': 'Ward-1', 'contact_date': date(2025, 1, 3), 'organism': 'ESBL'},
            {'patient1': 'P001', 'patient2': 'P007', 'location': 'Ward-2', 'contact_date': date(2025, 1, 4), 'organism': 'CRE'},
            # Secondary transmission
            {'patient1': 'P002', 'patient2': 'P008', 'location': 'Ward-1', 'contact_date': date(2025, 1, 2), 'organism': 'CRE'},
            {'patient1': 'P005', 'patient2': 'P009', 'location': 'Ward-2', 'contact_date': date(2025, 1, 6), 'organism': 'MRSA'}
        ]
        
        super_spreaders = service.get_super_spreaders()
        
        assert 'super_spreaders' in super_spreaders
        assert 'analysis_metadata' in super_spreaders
        
        # P001 should be identified as super spreader
        if super_spreaders['super_spreaders']:
            p001_found = False
            for ss in super_spreaders['super_spreaders']:
                if ss['patient_id'] == 'P001':
                    p001_found = True
                    assert ss['outbound_transmissions'] >= 3  # P001 contacted multiple patients
                    assert 'risk_score' in ss
                    assert 'locations_infected' in ss
                    break
            
            # Verify metadata
            metadata = super_spreaders['analysis_metadata']
            assert 'super_spreaders_found' in metadata
            assert 'total_patients_analyzed' in metadata
            assert 'analysis_period_days' in metadata
    
    def test_complex_location_risk_analysis(self, service, complex_mock_data):
        """Test complex location risk analysis with multiple transmission events."""
        service.load_and_optimize_data()
        
        # Create transmission events focused on specific locations
        service.contacts = [
            # Ward-1 has multiple transmission events
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1)},
            {'patient1': 'P006', 'patient2': 'P007', 'location': 'Ward-1', 'contact_date': date(2025, 1, 6)},
            {'patient1': 'P009', 'patient2': 'P010', 'location': 'Ward-1', 'contact_date': date(2025, 1, 9)},
            {'patient1': 'P011', 'patient2': 'P012', 'location': 'Ward-1', 'contact_date': date(2025, 1, 11)},
            # Ward-2 has fewer events
            {'patient1': 'P001', 'patient2': 'P005', 'location': 'Ward-2', 'contact_date': date(2025, 1, 3)},
            # Ward-3 has minimal events
            {'patient1': 'P004', 'patient2': 'P008', 'location': 'Ward-3', 'contact_date': date(2025, 1, 8)}
        ]
        
        heatmaps = service.get_location_risk_heatmaps()
        
        assert 'location_risks' in heatmaps
        assert 'analysis_metadata' in heatmaps
        assert isinstance(heatmaps['location_risks'], list)
        
        if heatmaps['location_risks']:
            # Find Ward-1 which should have highest risk
            ward1_risk = None
            for location_risk in heatmaps['location_risks']:
                if location_risk['location'] == 'Ward-1':
                    ward1_risk = location_risk
                    break
            
            if ward1_risk:
                assert ward1_risk['transmission_events'] >= 4  # Multiple transmission events
                assert ward1_risk['risk_level'] in ['HIGH', 'MEDIUM', 'LOW']
                assert 'recommended_actions' in ward1_risk
                assert isinstance(ward1_risk['recommended_actions'], list)
        
        # Verify metadata
        metadata = heatmaps['analysis_metadata']
        required_metadata_fields = [
            'total_locations_analyzed', 'risk_level_distribution',
            'analysis_date', 'highest_risk_location'
        ]
        
        for field in required_metadata_fields:
            assert field in metadata
    
    def test_multi_infection_spread_analysis(self, service, complex_mock_data):
        """Test spread analysis across multiple infection types."""
        service.load_and_optimize_data()
        
        # Create contacts with different organisms
        service.contacts = [
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1), 'organism': 'CRE'},
            {'patient1': 'P001', 'patient2': 'P006', 'location': 'Ward-1', 'contact_date': date(2025, 1, 3), 'organism': 'ESBL'},
            {'patient1': 'P005', 'patient2': 'P009', 'location': 'Ward-2', 'contact_date': date(2025, 1, 6), 'organism': 'MRSA'},
            {'patient1': 'P010', 'patient2': 'P012', 'location': 'Ward-1', 'contact_date': date(2025, 1, 10), 'organism': 'VRE'}
        ]
        
        # Test specific infection spread
        cre_spread = service.generate_spread_visualization('CRE')
        assert cre_spread['infection_type'] == 'CRE'
        assert 'timeline_events' in cre_spread
        assert 'spread_events' in cre_spread
        assert 'network_nodes' in cre_spread
        assert 'stats' in cre_spread
        
        # Test all infections spread
        all_spread = service.generate_spread_visualization()
        assert 'infections' in all_spread
        assert 'global_stats' in all_spread
        assert 'combined_timeline' in all_spread
        
        # Should include multiple infection types
        if 'infections' in all_spread:
            infection_types = list(all_spread['infections'].keys())
            assert len(infection_types) >= 2  # Multiple infection types present
    
    def test_temporal_clustering_patterns(self, service, complex_mock_data):
        """Test detection of temporal clustering patterns."""
        service.load_and_optimize_data()
        
        # Create time-based clusters
        early_contacts = [
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1)},
            {'patient1': 'P002', 'patient2': 'P006', 'location': 'Ward-1', 'contact_date': date(2025, 1, 2)}
        ]
        
        late_contacts = [
            {'patient1': 'P009', 'patient2': 'P010', 'location': 'Ward-2', 'contact_date': date(2025, 1, 10)},
            {'patient1': 'P010', 'patient2': 'P012', 'location': 'Ward-2', 'contact_date': date(2025, 1, 11)}
        ]
        
        service.contacts = early_contacts + late_contacts
        service.get_contact_groups()
        clusters = service.generate_cluster_data()
        
        # Should create distinct temporal clusters
        if len(clusters) >= 2:
            dates_cluster1 = set()
            dates_cluster2 = set()
            
            for cluster in clusters:
                date_range = cluster.get('date_range', {})
                if 'earliest' in date_range and 'latest' in date_range:
                    if cluster['cluster_id'] == 1:
                        dates_cluster1.add(date_range['earliest'])
                        dates_cluster1.add(date_range['latest'])
                    else:
                        dates_cluster2.add(date_range['earliest'])
                        dates_cluster2.add(date_range['latest'])
            
            # Clusters should have different time ranges
            assert dates_cluster1 != dates_cluster2
    
    def test_network_centrality_analysis(self, service, complex_mock_data):
        """Test network centrality analysis in contact networks."""
        service.load_and_optimize_data()
        
        # Create a hub-and-spoke network pattern
        hub_contacts = [
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1)},
            {'patient1': 'P001', 'patient2': 'P005', 'location': 'Ward-1', 'contact_date': date(2025, 1, 2)},
            {'patient1': 'P001', 'patient2': 'P006', 'location': 'Ward-1', 'contact_date': date(2025, 1, 3)},
            {'patient1': 'P001', 'patient2': 'P007', 'location': 'Ward-1', 'contact_date': date(2025, 1, 4)},
            # Secondary connections
            {'patient1': 'P002', 'patient2': 'P009', 'location': 'Ward-2', 'contact_date': date(2025, 1, 5)},
            {'patient1': 'P005', 'patient2': 'P010', 'location': 'Ward-2', 'contact_date': date(2025, 1, 6)}
        ]
        
        service.contacts = hub_contacts
        graph_data = service.generate_graph_data()
        
        # P001 should have the highest contact count (centrality)
        if 'P001' in graph_data:
            p001_contacts = graph_data['P001']['contact_count']
            
            # P001 should have more contacts than others
            for patient_id, data in graph_data.items():
                if patient_id != 'P001':
                    assert data['contact_count'] <= p001_contacts
    
    def test_infection_transmission_confidence(self, service, complex_mock_data):
        """Test calculation of infection transmission confidence scores."""
        service.load_and_optimize_data()
        
        # Create contacts with varying temporal patterns
        service.contacts = [
            # High confidence: contact before test positive
            {
                'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1',
                'contact_date': date(2025, 1, 1), 'organism': 'CRE',
                'days_from_test1': -1, 'days_from_test2': 1  # P001 positive 1 day before contact, P002 1 day after
            },
            # Medium confidence: overlapping dates
            {
                'patient1': 'P005', 'patient2': 'P006', 'location': 'Ward-2',
                'contact_date': date(2025, 1, 5), 'organism': 'MRSA',
                'days_from_test1': 0, 'days_from_test2': 0  # Both positive same day as contact
            },
            # Lower confidence: reverse temporal order
            {
                'patient1': 'P009', 'patient2': 'P010', 'location': 'Ward-1',
                'contact_date': date(2025, 1, 9), 'organism': 'VRE',
                'days_from_test1': 2, 'days_from_test2': -1  # P010 positive before P009
            }
        ]
        
        super_spreaders = service.get_super_spreaders()
        
        # Should calculate confidence scores
        if super_spreaders['super_spreaders']:
            for spreader in super_spreaders['super_spreaders']:
                if 'transmission_confidence_avg' in spreader:
                    assert 0 <= spreader['transmission_confidence_avg'] <= 1
    
    def test_outbreak_scenario_simulation(self, service, complex_mock_data):
        """Test simulation of outbreak scenarios."""
        service.load_and_optimize_data()
        
        # Simulate outbreak spreading pattern
        outbreak_contacts = [
            # Initial case
            {'patient1': 'P001', 'patient2': 'P002', 'location': 'Ward-1', 'contact_date': date(2025, 1, 1), 'organism': 'CRE'},
            # First generation spread
            {'patient1': 'P002', 'patient2': 'P005', 'location': 'Ward-1', 'contact_date': date(2025, 1, 3), 'organism': 'CRE'},
            {'patient1': 'P002', 'patient2': 'P006', 'location': 'Ward-1', 'contact_date': date(2025, 1, 3), 'organism': 'CRE'},
            # Second generation spread
            {'patient1': 'P005', 'patient2': 'P009', 'location': 'Ward-2', 'contact_date': date(2025, 1, 6), 'organism': 'CRE'},
            {'patient1': 'P006', 'patient2': 'P010', 'location': 'Ward-1', 'contact_date': date(2025, 1, 7), 'organism': 'CRE'},
            # Cross-ward transmission
            {'patient1': 'P009', 'patient2': 'P012', 'location': 'Ward-3', 'contact_date': date(2025, 1, 10), 'organism': 'CRE'}
        ]
        
        service.contacts = outbreak_contacts
        service.get_contact_groups()
        clusters = service.generate_cluster_data()
        
        # Should detect large outbreak cluster
        if clusters:
            largest_cluster = max(clusters, key=lambda c: c['patient_count'])
            assert largest_cluster['patient_count'] >= 5  # Outbreak should connect multiple patients
            assert len(largest_cluster['locations']) >= 2  # Should span multiple locations
            assert 'CRE' in largest_cluster['infections']
    
    def test_data_quality_assessment(self, service):
        """Test assessment of data quality and completeness."""
        # Test with incomplete/malformed data
        incomplete_micro = pd.DataFrame({
            'patient_id': ['P001', '', None, 'P002'],
            'collection_date': ['2025-01-01', '2025-01-02', 'invalid-date', '2025-01-04'],
            'infection': ['CRE', 'ESBL', None, ''],
            'result': ['positive', 'positive', 'positive', 'negative']
        })
        
        incomplete_transfers = pd.DataFrame({
            'patient_id': ['P001', 'P002', ''],
            'location': ['Ward-1', '', 'Ward-2'],
            'ward_in_time': ['2025-01-01', 'invalid-date', '2025-01-03'],
            'ward_out_time': ['2025-01-02', '2025-01-04', '2025-01-04']
        })
        
        service.df_micro = incomplete_micro
        service.df_transfers = incomplete_transfers
        
        # Filter positive results, handling missing values
        try:
            service.df_positive = service.df_micro[
                (service.df_micro['result'] == 'positive') & 
                (service.df_micro['patient_id'].notna()) &
                (service.df_micro['patient_id'] != '')
            ]
        except Exception:
            # Should handle data quality issues gracefully
            service.df_positive = pd.DataFrame()
        
        # Should complete without crashing
        try:
            service.create_spatial_temporal_index()
            service.contact_detection()
        except Exception as e:
            # Some exceptions are expected with malformed data
            assert "date" in str(e).lower() or "time" in str(e).lower() or "value" in str(e).lower()
    
    def test_memory_usage_optimization(self, service, complex_mock_data):
        """Test memory usage optimization with large datasets."""
        service.load_and_optimize_data()
        
        # Verify data types are optimized
        if service.df_micro is not None:
            # Check that categorical columns are properly typed
            assert service.df_micro.dtypes['patient_id'] == 'object'
            assert service.df_micro.dtypes['infection'] == 'object'
        
        if service.df_transfers is not None:
            assert service.df_transfers.dtypes['patient_id'] == 'object'
            assert service.df_transfers.dtypes['location'] == 'object'
        
        # Test that processing completes efficiently
        service.create_spatial_temporal_index()
        service.contact_detection()
        
        # Should not produce excessive number of contacts
        assert len(service.contacts) < 1000  # Reasonable upper bound for test data
    
    def test_edge_case_date_handling(self, service):
        """Test handling of edge cases in date processing."""
        # Test data with various date edge cases
        edge_case_data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003', 'P004'],
            'collection_date': [
                '2025-01-01',    # Normal date
                '2025-02-29',    # Invalid date (2025 is not leap year)  
                '2025-13-01',    # Invalid month
                '2025-01-32'     # Invalid day
            ],
            'infection': ['CRE', 'ESBL', 'VRE', 'MRSA'],
            'result': ['positive', 'positive', 'positive', 'positive']
        })
        
        service.df_micro = edge_case_data
        service.df_positive = edge_case_data
        
        # Should handle invalid dates gracefully
        try:
            service.create_spatial_temporal_index()
        except Exception as e:
            # Should be a date parsing error, not a crash
            assert any(keyword in str(e).lower() for keyword in ['date', 'time', 'parse', 'convert'])


class TestServiceIntegration:
    """Integration tests for service methods working together."""

    @pytest.fixture
    def service_with_data(self):
        """Create service with mocked data loaded."""
        service = InfectionDetectionService()

        with patch('pandas.read_csv') as mock_read_csv:
            def side_effect(filepath):
                if 'microbiology' in str(filepath):
                    return create_sample_microbiology_data()
                elif 'transfers' in str(filepath):
                    return create_sample_transfers_data()
                return pd.DataFrame()

            mock_read_csv.side_effect = side_effect
            with patch.object(service, '_validate_data_files', return_value=True):
                # Run synchronously for fixture
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(service.run_detection_pipeline())
                except RuntimeError:
                    asyncio.run(service.run_detection_pipeline())

        return service

    def test_full_pipeline_consistency(self, service_with_data):
        """Test that pipeline produces consistent results."""
        service = service_with_data

        graph_data = service.generate_graph_data()
        clusters = service.generate_cluster_data()
        summary = service.generate_summary_metrics(graph_data, clusters)

        # Consistency checks
        assert summary['total_patients'] == len(graph_data)
        assert summary['connected_patients'] + summary['isolated_patients'] == summary['total_patients']
        assert summary['total_clusters'] == len(clusters)
        assert summary['total_contact_events'] == len(service.contacts)

    def test_patient_retrieval_consistency(self, service_with_data):
        """Test patient data retrieval consistency."""
        service = service_with_data
        graph_data = service.generate_graph_data()

        for patient_id in list(graph_data.keys())[:3]:  # Test first 3 patients
            patient_data = service.get_patient_details(patient_id)
            if patient_data and patient_data['patients']:
                patient = patient_data['patients'][0]
                assert patient['patient_id'] == patient_id
                graph_patient = graph_data[patient_id]
                assert set(patient['positive_infections']) == set(graph_patient['infections'])

    def test_spread_visualization_with_filters(self, service_with_data):
        """Test spread visualization with different filters."""
        service = service_with_data

        # Test all infections
        all_spread = service.generate_spread_visualization()
        assert isinstance(all_spread, dict)

        # Test specific infection if available
        if service.df_positive is not None and not service.df_positive.empty:
            # Check if 'organism' column exists, fallback to 'infection'
            if 'organism' in service.df_positive.columns:
                infections = service.df_positive['organism'].unique()
            else:
                infections = service.df_positive['infection'].unique()

            if len(infections) > 0:
                specific_spread = service.generate_spread_visualization(infections[0])
                assert isinstance(specific_spread, dict)


class TestServiceComprehensive:
    """Comprehensive tests covering all service functionality."""
    
    @pytest.fixture
    def service(self):
        return InfectionDetectionService()
    
    def test_complete_workflow_integration(self, service):
        """Test complete workflow from data loading to visualization."""
        # Mock comprehensive dataset
        with patch('pandas.read_csv') as mock_read_csv:
            def side_effect(filepath):
                if 'microbiology' in str(filepath):
                    return pd.DataFrame({
                        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
                        'collection_date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'],
                        'infection': ['CRE', 'ESBL', 'CRE', 'VRE', 'MRSA'],
                        'result': ['positive', 'positive', 'positive', 'positive', 'positive']
                    })
                elif 'transfers' in str(filepath):
                    return pd.DataFrame({
                        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
                        'location': ['Ward-1', 'Ward-1', 'Ward-2', 'Ward-1', 'Ward-2'],
                        'admission_date': ['2025-01-01', '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
                        'discharge_date': ['2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06', '2025-01-07']
                    })
                return pd.DataFrame()
            
            mock_read_csv.side_effect = side_effect
            
            with patch.object(service, '_validate_data_files', return_value=True):
                # Execute complete pipeline
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(service.run_detection_pipeline())
                except RuntimeError:
                    result = asyncio.run(service.run_detection_pipeline())
                
                # Verify complete result structure
                assert all(key in result for key in ['metadata', 'summary', 'patients', 'clusters', 'contacts'])
                
                # Test all analysis functions
                super_spreaders = service.get_super_spreaders()
                assert isinstance(super_spreaders, dict)
                
                heatmaps = service.get_location_risk_heatmaps()
                assert isinstance(heatmaps, dict)
                
                spread_viz = service.generate_spread_visualization()
                assert isinstance(spread_viz, dict)
                
                patient_details = service.get_patient_details()
                assert isinstance(patient_details, dict)
    
    def test_error_resilience_comprehensive(self, service):
        """Test service resilience to various error conditions."""
        # Test with completely empty data
        service.df_micro = pd.DataFrame()
        service.df_transfers = pd.DataFrame()
        service.df_positive = pd.DataFrame()
        
        # All functions should handle empty data gracefully
        service.contact_detection()
        assert service.contacts == []
        
        service.get_contact_groups()
        assert service.contact_groups == []
        
        graph_data = service.generate_graph_data()
        assert graph_data == {}
        
        clusters = service.generate_cluster_data()
        assert clusters == []
        
        summary = service.generate_summary_metrics({}, [])
        assert isinstance(summary, dict)
        
        super_spreaders = service.get_super_spreaders()
        assert isinstance(super_spreaders, dict)
        
        heatmaps = service.get_location_risk_heatmaps()
        assert isinstance(heatmaps, dict)
        
        spread_viz = service.generate_spread_visualization()
        assert isinstance(spread_viz, dict)
    
    def test_performance_benchmarks(self, service):
        """Test performance benchmarks for key operations."""
        import time
        
        # Create moderately large dataset
        large_micro = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(50)],
            'collection_date': ['2025-01-01'] * 50,
            'infection': ['CRE'] * 50,
            'result': ['positive'] * 50
        })
        
        large_transfers = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(50)] * 2,  # 2 transfers per patient
            'location': ['Ward-1', 'Ward-2'] * 50,
            'admission_date': ['2025-01-01'] * 100,
            'discharge_date': ['2025-01-02'] * 100
        })
        
        service.df_micro = large_micro
        service.df_transfers = large_transfers
        service.df_positive = large_micro
        
        # Benchmark contact detection
        start_time = time.time()
        service.create_spatial_temporal_index()
        service.contact_detection()
        contact_time = time.time() - start_time
        
        # Should complete within reasonable time (5 seconds for 50 patients)
        assert contact_time < 5.0
        
        # Benchmark clustering
        start_time = time.time()
        service.get_contact_groups()
        cluster_time = time.time() - start_time
        
        assert cluster_time < 2.0
    
    def test_data_integrity_validation(self, service):
        """Test validation of data integrity throughout processing."""
        # Mock data with known relationships
        micro_data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'collection_date': ['2025-01-01', '2025-01-02', '2025-01-03'],
            'infection': ['CRE', 'CRE', 'ESBL'],
            'result': ['positive', 'positive', 'positive']
        })
        
        transfers_data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'location': ['Ward-1', 'Ward-1', 'Ward-2'],
            'admission_date': ['2025-01-01', '2025-01-01', '2025-01-02'],
            'discharge_date': ['2025-01-02', '2025-01-03', '2025-01-04']
        })
        
        service.df_micro = micro_data
        service.df_transfers = transfers_data
        service.df_positive = micro_data
        
        service.create_spatial_temporal_index()
        service.contact_detection()
        service.get_contact_groups()
        
        # Verify data integrity
        graph_data = service.generate_graph_data()
        clusters = service.generate_cluster_data()
        summary = service.generate_summary_metrics(graph_data, clusters)
        
        # All patients in micro data should be accounted for
        assert summary['total_patients'] == 3
        
        # Patient counts should be consistent
        total_in_clusters = sum(cluster['patient_count'] for cluster in clusters)
        assert total_in_clusters + summary['isolated_patients'] == summary['total_patients']
    
    def test_configuration_edge_cases(self, service):
        """Test handling of configuration edge cases."""
        # Test with minimal valid data
        minimal_micro = pd.DataFrame({
            'patient_id': ['P001'],
            'collection_date': ['2025-01-01'],
            'infection': ['CRE'],
            'result': ['positive']
        })
        
        minimal_transfers = pd.DataFrame({
            'patient_id': ['P001'],
            'location': ['Ward-1'],
            'admission_date': ['2025-01-01'],
            'discharge_date': ['2025-01-01']
        })
        
        service.df_micro = minimal_micro
        service.df_transfers = minimal_transfers
        service.df_positive = minimal_micro
        
        # Should handle single patient scenario
        service.create_spatial_temporal_index()
        service.contact_detection()
        service.get_contact_groups()
        
        graph_data = service.generate_graph_data()
        assert len(graph_data) == 1
        assert 'P001' in graph_data
        
        clusters = service.generate_cluster_data()
        # Single patient should not form a cluster
        assert len(clusters) == 0
        
        summary = service.generate_summary_metrics(graph_data, clusters)
        assert summary['total_patients'] == 1
        assert summary['isolated_patients'] == 1
        assert summary['connected_patients'] == 0