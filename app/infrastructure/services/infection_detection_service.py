#!/usr/bin/env python3
"""
Infection Spreading Detection Service

A service for detecting clusters of infections in hospital records.
Integrates with FastAPI application structure.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numba
import numpy as np
import pandas as pd

from app.core.config import microbiology_file, transfers_file

logger = logging.getLogger(__name__)


# ---------------- Union-Find ----------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def get_groups(self) -> list[list[int]]:
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return [g for g in groups.values() if len(g) > 1]


# ---------------- Numba kernels ----------------
@numba.jit(nopython=True)
def date_overlap(starts1, ends1, starts2, ends2):
    n1, n2 = len(starts1), len(starts2)
    overlaps = np.zeros((n1, n2), dtype=np.bool_)
    overlap_starts = np.zeros((n1, n2), dtype=np.int32)
    overlap_ends = np.zeros((n1, n2), dtype=np.int32)
    for i in range(n1):
        for j in range(n2):
            start = starts1[i] if starts1[i] >= starts2[j] else starts2[j]
            end = ends1[i] if ends1[i] <= ends2[j] else ends2[j]
            if start <= end:
                overlaps[i, j] = True
                overlap_starts[i, j] = start
                overlap_ends[i, j] = end
    return overlaps, overlap_starts, overlap_ends


@numba.jit(nopython=True)
def check_within_window(contact_dates, test_dates, window=14):
    valid = np.zeros(len(contact_dates), dtype=np.bool_)
    for i in range(len(contact_dates)):
        cd = contact_dates[i]
        for j in range(len(test_dates)):
            if abs(cd - test_dates[j]) <= window:
                valid[i] = True
                break
    return valid


# ---------------- Service ----------------
class InfectionDetectionService:
    """
    Service for detecting infection clusters in hospital data.

    Detects clusters of infections based on:
    - Two or more patients with positive tests for the same organism
    - Linked by spatial-temporal contact events (same location, overlapping days)
    - Contact within ±14 days of both patients' test dates
    - Transitive clustering (A-B-C chains)
    """

    def __init__(
        self,
        window_days: int = 14,
        date_origin: datetime | None = None,
    ):
        self.window_days = window_days
        self.date_origin = (
            date_origin.date()
            if isinstance(date_origin, datetime)
            else (date_origin if date_origin else datetime(2025, 4, 1).date())
        )

        # Data paths
        self.microbiology_file = microbiology_file
        self.transfers_file = transfers_file

        # DataFrames
        self.df_micro = None
        self.df_transfers = None
        self.df_positive = None

        # Index maps
        self.patient_to_idx = {}
        self.idx_to_patient = {}
        self.location_to_idx = {}
        self.idx_to_location = {}

        # Presence partitions
        self.df_transfers_positive = None
        self.location_groups = {}

        # Precomputed test maps
        self.patient_tests_any = {}
        self.patient_tests_by_org = {}
        self.patient_orgs = {}

        # Outputs
        self.contacts = []
        self.contact_groups = []

    def _validate_data_files(self) -> bool:
        """Validate that required data files exist."""
        if not self.microbiology_file.exists():
            logger.error(f"Microbiology file not found: {self.microbiology_file}")
            return False
        if not self.transfers_file.exists():
            logger.error(f"Transfers file not found: {self.transfers_file}")
            return False
        return True

    def load_and_optimize_data(self) -> None:
        """Load and preprocess hospital data."""
        if not self._validate_data_files():
            raise FileNotFoundError("Required data files are missing")

        logger.info("Loading microbiology data...")
        self.df_micro = pd.read_csv(self.microbiology_file)
        self.df_micro["collection_date"] = pd.to_datetime(
            self.df_micro["collection_date"]
        )
        self.df_micro["result"] = (
            self.df_micro["result"].astype(str).str.lower().str.strip()
        )
        self.df_positive = self.df_micro[self.df_micro["result"] == "positive"].copy()

        logger.info("Loading transfers data...")
        self.df_transfers = pd.read_csv(self.transfers_file)
        self.df_transfers["ward_in_time"] = pd.to_datetime(
            self.df_transfers["ward_in_time"]
        )
        self.df_transfers["ward_out_time"] = pd.to_datetime(
            self.df_transfers["ward_out_time"]
        )

        # Convert dates to day offsets for efficient computation
        self.df_positive["test_day"] = (
            self.df_positive["collection_date"].dt.date.apply(
                lambda x: (x - self.date_origin).days
            )
        ).astype(np.int32)
        self.df_transfers["start_day"] = (
            self.df_transfers["ward_in_time"].dt.date.apply(
                lambda x: (x - self.date_origin).days
            )
        ).astype(np.int32)
        self.df_transfers["end_day"] = (
            self.df_transfers["ward_out_time"].dt.date.apply(
                lambda x: (x - self.date_origin).days
            )
        ).astype(np.int32)

        # Create index mappings
        all_patients = set(self.df_positive["patient_id"]) | set(
            self.df_transfers["patient_id"]
        )
        self.patient_to_idx = {p: i for i, p in enumerate(sorted(all_patients))}
        self.idx_to_patient = {i: p for p, i in self.patient_to_idx.items()}

        all_locations = set(self.df_transfers["location"])
        self.location_to_idx = {loc: i for i, loc in enumerate(sorted(all_locations))}
        self.idx_to_location = {i: loc for loc, i in self.location_to_idx.items()}

        # Add index columns
        self.df_positive["patient_idx"] = self.df_positive["patient_id"].map(
            self.patient_to_idx
        )
        self.df_transfers["patient_idx"] = self.df_transfers["patient_id"].map(
            self.patient_to_idx
        )
        self.df_transfers["location_idx"] = self.df_transfers["location"].map(
            self.location_to_idx
        )

        # Filter transfers to only positive patients
        pos_idx = set(self.df_positive["patient_idx"])
        self.df_transfers_positive = self.df_transfers[
            self.df_transfers["patient_idx"].isin(pos_idx)
        ].copy()

        # Precompute test date maps
        tmp_any = defaultdict(list)
        for r in self.df_positive.itertuples(index=False):
            tmp_any[r.patient_idx].append(int(r.test_day))
        self.patient_tests_any = {
            pid: np.array(days, dtype=np.int32) for pid, days in tmp_any.items()
        }

        tmp_org = defaultdict(list)
        patient_orgs = defaultdict(set)
        for r in self.df_positive.itertuples(index=False):
            key = (r.patient_idx, r.infection)
            tmp_org[key].append(int(r.test_day))
            patient_orgs[r.patient_idx].add(r.infection)
        self.patient_tests_by_org = {
            k: np.array(v, dtype=np.int32) for k, v in tmp_org.items()
        }
        self.patient_orgs = {pid: set(orgs) for pid, orgs in patient_orgs.items()}

        logger.info(
            f"Loaded {len(self.df_positive)} positive tests for {len(all_patients)} patients"
        )

    def create_spatial_temporal_index(self):
        """Create location-based groupings for efficient contact detection."""
        self.location_groups = {}
        for loc_idx in self.df_transfers_positive["location_idx"].unique():
            sub = self.df_transfers_positive[
                self.df_transfers_positive["location_idx"] == loc_idx
            ].copy()
            sub = sub.sort_values("start_day")
            self.location_groups[loc_idx] = sub

        logger.info(
            f"Created spatial-temporal index for {len(self.location_groups)} locations"
        )

    def contact_detection(self):
        """Detect contact events between patients with shared organisms."""
        contacts_out = []

        for loc_idx, loc_df in self.location_groups.items():
            if len(loc_df) < 2:
                continue

            patients = loc_df["patient_idx"].values
            starts = loc_df["start_day"].values
            ends = loc_df["end_day"].values
            n = len(loc_df)

            overlaps, ostarts, oends = date_overlap(starts, ends, starts, ends)

            for i in range(n):
                for j in range(i + 1, n):
                    if not overlaps[i, j]:
                        continue

                    p1, p2 = int(patients[i]), int(patients[j])
                    if p1 == p2:
                        continue
                    if (
                        p1 not in self.patient_tests_any
                        or p2 not in self.patient_tests_any
                    ):
                        continue

                    cd_start, cd_end = int(ostarts[i, j]), int(oends[i, j])
                    contact_dates = np.arange(cd_start, cd_end + 1, dtype=np.int32)

                    shared_orgs = self.patient_orgs.get(p1, set()).intersection(
                        self.patient_orgs.get(p2, set())
                    )
                    if not shared_orgs:
                        continue

                    for org in shared_orgs:
                        t1 = self.patient_tests_by_org.get((p1, org))
                        t2 = self.patient_tests_by_org.get((p2, org))
                        if t1 is None or t2 is None:
                            continue

                        v1 = check_within_window(contact_dates, t1, self.window_days)
                        v2 = check_within_window(contact_dates, t2, self.window_days)
                        mask = v1 & v2

                        for cd in contact_dates[mask]:
                            d1 = int(cd - t1[np.argmin(np.abs(t1 - cd))])
                            d2 = int(cd - t2[np.argmin(np.abs(t2 - cd))])
                            contacts_out.append(
                                {
                                    "patient1": self.idx_to_patient[p1],
                                    "patient2": self.idx_to_patient[p2],
                                    "location": self.idx_to_location[loc_idx],
                                    "organism": org,
                                    "contact_date": self.date_origin
                                    + timedelta(days=int(cd)),
                                    "days_from_test1": d1,
                                    "days_from_test2": d2,
                                }
                            )

        self.contacts = contacts_out
        logger.info(f"Detected {len(contacts_out)} contact events")

    def get_contact_groups(self):
        """Generate infection clusters using Union-Find algorithm."""
        if not self.contacts:
            self.contact_groups = []
            return

        pats = sorted(
            {c["patient1"] for c in self.contacts}
            | {c["patient2"] for c in self.contacts}
        )
        name_to_idx = {p: i for i, p in enumerate(pats)}
        uf = UnionFind(len(pats))

        for c in self.contacts:
            uf.union(name_to_idx[c["patient1"]], name_to_idx[c["patient2"]])

        idx_groups = uf.get_groups()
        self.contact_groups = [[pats[i] for i in grp] for grp in idx_groups]

        logger.info(f"Generated {len(self.contact_groups)} infection clusters")

    def generate_graph_data(self) -> dict[str, Any]:
        """Generate patient contact graph data."""
        patient_contacts = defaultdict(set)
        contact_details = defaultdict(list)

        for contact in self.contacts:
            patient1, patient2 = contact["patient1"], contact["patient2"]
            patient_contacts[patient1].add(patient2)
            patient_contacts[patient2].add(patient1)

            # Store contact details
            contact_details[patient1].append(
                {
                    "with_patient": patient2,
                    "location": contact["location"],
                    "date": contact["contact_date"].strftime("%Y-%m-%d"),
                }
            )
            contact_details[patient2].append(
                {
                    "with_patient": patient1,
                    "location": contact["location"],
                    "date": contact["contact_date"].strftime("%Y-%m-%d"),
                }
            )

        # Create patient info mapping
        patient_info = {}
        for _, row in self.df_positive.iterrows():
            patient_id = row["patient_id"]
            if patient_id not in patient_info:
                patient_info[patient_id] = {
                    "patient_id": patient_id,
                    "infections": [],
                    "test_dates": [],
                }

            patient_info[patient_id]["infections"].append(row["infection"])
            patient_info[patient_id]["test_dates"].append(
                row["collection_date"].strftime("%Y-%m-%d")
            )

        # Build the graph format
        graph_data = {}
        for patient_id in patient_info.keys():
            contacted_patients = list(patient_contacts.get(patient_id, set()))

            graph_data[patient_id] = {
                "contacts": contacted_patients,
                "contact_count": len(contacted_patients),
                "infections": list(set(patient_info[patient_id]["infections"])),
                "primary_infection": patient_info[patient_id]["infections"][0],
                "test_dates": patient_info[patient_id]["test_dates"],
                "contact_details": contact_details.get(patient_id, []),
            }

        return graph_data

    def generate_cluster_data(self) -> list[dict[str, Any]]:
        """Generate cluster/group data."""
        clusters = []

        for i, group in enumerate(self.contact_groups):
            # Get cluster statistics
            cluster_infections = []
            cluster_test_dates = []
            cluster_patients = []

            for patient_id in group:
                # Find patient tests
                patient_tests = []
                for _, row in self.df_positive.iterrows():
                    if row["patient_id"] == patient_id:
                        patient_tests.append(
                            {
                                "infection": row["infection"],
                                "collection_date": row["collection_date"],
                            }
                        )

                patient_data = {
                    "patient_id": patient_id,
                    "infections": [test["infection"] for test in patient_tests],
                    "test_dates": [
                        test["collection_date"].strftime("%Y-%m-%d")
                        for test in patient_tests
                    ],
                }

                cluster_patients.append(patient_data)
                cluster_infections.extend([test["infection"] for test in patient_tests])
                cluster_test_dates.extend(
                    [test["collection_date"].date() for test in patient_tests]
                )

            # Cluster contacts
            cluster_contacts = [
                contact
                for contact in self.contacts
                if contact["patient1"] in group and contact["patient2"] in group
            ]

            # Cluster locations
            cluster_locations = list(
                set([contact["location"] for contact in cluster_contacts])
            )

            cluster_data = {
                "cluster_id": i + 1,
                "patients": cluster_patients,
                "patient_count": len(group),
                "contact_count": len(cluster_contacts),
                "infections": list(set(cluster_infections)),
                "infection_counts": {
                    infection: cluster_infections.count(infection)
                    for infection in set(cluster_infections)
                },
                "locations": cluster_locations,
                "date_range": {
                    "earliest": min(cluster_test_dates).isoformat()
                    if cluster_test_dates
                    else None,
                    "latest": max(cluster_test_dates).isoformat()
                    if cluster_test_dates
                    else None,
                },
                "contacts": [
                    {
                        "patient1": contact["patient1"],
                        "patient2": contact["patient2"],
                        "location": contact["location"],
                        "contact_date": contact["contact_date"].strftime("%Y-%m-%d"),
                    }
                    for contact in cluster_contacts
                ],
            }

            clusters.append(cluster_data)

        return clusters

    def get_patient_details(self, patient_id: str | None = None) -> dict[str, Any]:
        """Get detailed patient information including tests and transfers."""
        if self.df_micro is None or self.df_transfers is None:
            # Load data if not already loaded
            self.load_and_optimize_data()

        patient_details = []
        unique_infections = set()
        total_positive_tests = 0

        # Get all patients (or specific patient if provided)
        if patient_id:
            patients_to_process = (
                [patient_id]
                if patient_id
                in set(self.df_micro["patient_id"])
                | set(self.df_transfers["patient_id"])
                else []
            )
            if not patients_to_process:
                return {
                    "patients": [],
                    "total_patients": 0,
                    "patients_with_positive_tests": 0,
                    "total_positive_tests": 0,
                    "unique_infections": [],
                }
        else:
            patients_to_process = sorted(
                set(self.df_micro["patient_id"]) | set(self.df_transfers["patient_id"])
            )

        for pid in patients_to_process:
            # Get test cases for this patient
            patient_tests = self.df_micro[self.df_micro["patient_id"] == pid]
            test_cases = []
            positive_infections = []

            for _, test in patient_tests.iterrows():
                test_case = {
                    "patient_id": pid,
                    "collection_date": test["collection_date"].strftime("%Y-%m-%d"),
                    "infection": test["infection"],
                    "result": test["result"],
                }
                test_cases.append(test_case)

                if test["result"].lower().strip() == "positive":
                    positive_infections.append(test["infection"])
                    unique_infections.add(test["infection"])
                    total_positive_tests += 1

            # Get transfers for this patient
            patient_transfers = self.df_transfers[
                self.df_transfers["patient_id"] == pid
            ]
            transfers = []

            for _, transfer in patient_transfers.iterrows():
                # Calculate duration
                duration = (
                    transfer["ward_out_time"] - transfer["ward_in_time"]
                ).total_seconds() / 3600

                transfer_record = {
                    "patient_id": pid,
                    "location": transfer["location"],
                    "ward_in_time": transfer["ward_in_time"].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "ward_out_time": transfer["ward_out_time"].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "duration_hours": round(duration, 2),
                }
                transfers.append(transfer_record)

            # Calculate summary stats for this patient
            positive_test_dates = [
                test["collection_date"]
                for test in test_cases
                if test["result"].lower().strip() == "positive"
            ]
            all_test_dates = [test["collection_date"] for test in test_cases]

            patient_detail = {
                "patient_id": pid,
                "test_cases": test_cases,
                "transfers": transfers,
                "positive_infections": list(set(positive_infections)),
                "total_tests": len(test_cases),
                "total_transfers": len(transfers),
                "first_positive_date": min(positive_test_dates)
                if positive_test_dates
                else None,
                "last_test_date": max(all_test_dates) if all_test_dates else None,
            }

            patient_details.append(patient_detail)

        patients_with_positive = sum(
            1 for p in patient_details if p["positive_infections"]
        )

        return {
            "patients": patient_details,
            "total_patients": len(patient_details),
            "patients_with_positive_tests": patients_with_positive,
            "total_positive_tests": total_positive_tests,
            "unique_infections": sorted(list(unique_infections)),
        }

    def generate_spread_visualization(
        self, infection_type: str | None = None
    ) -> dict[str, Any]:
        """Generate data for spread visualization by infection type."""
        if self.df_micro is None or not self.contacts:
            # Ensure data is loaded and analyzed
            self.load_and_optimize_data()
            self.create_spatial_temporal_index()
            self.contact_detection()
            self.get_contact_groups()

        if infection_type:
            return self._generate_single_infection_spread(infection_type)
        else:
            return self._generate_all_infections_spread()

    def _generate_single_infection_spread(self, infection_type: str) -> dict[str, Any]:
        """Generate spread visualization data for a specific infection."""
        # Filter contacts for this infection
        infection_contacts = [
            c for c in self.contacts if c["organism"] == infection_type
        ]

        if not infection_contacts:
            return {
                "infection_type": infection_type,
                "timeline_events": [],
                "spread_events": [],
                "network_nodes": [],
                "network_edges": [],
                "date_range": {"earliest": None, "latest": None},
                "stats": {
                    "total_patients": 0,
                    "total_contacts": 0,
                    "total_spread_events": 0,
                },
            }

        # Get patients with this infection
        infection_patients = set()
        for contact in infection_contacts:
            infection_patients.add(contact["patient1"])
            infection_patients.add(contact["patient2"])

        # Generate timeline events
        timeline_events = []
        all_dates = []

        # Add positive test events
        positive_tests = self.df_positive[
            (self.df_positive["infection"] == infection_type)
            & (self.df_positive["patient_id"].isin(infection_patients))
        ]

        for _, test in positive_tests.iterrows():
            date_str = test["collection_date"].strftime("%Y-%m-%d")
            all_dates.append(test["collection_date"])

            timeline_events.append(
                {
                    "date": date_str,
                    "event_type": "positive_test",
                    "patient_id": test["patient_id"],
                    "infection": infection_type,
                    "location": None,
                    "related_patient": None,
                    "details": {"test_result": "positive"},
                }
            )

        # Add contact events
        for contact in infection_contacts:
            date_str = contact["contact_date"].strftime("%Y-%m-%d")
            # Convert to datetime if it's a date object
            if hasattr(contact["contact_date"], "date"):
                all_dates.append(contact["contact_date"])
            else:
                all_dates.append(
                    datetime.combine(contact["contact_date"], datetime.min.time())
                )

            timeline_events.append(
                {
                    "date": date_str,
                    "event_type": "contact",
                    "patient_id": contact["patient1"],
                    "infection": None,
                    "location": contact["location"],
                    "related_patient": contact["patient2"],
                    "details": {
                        "days_from_test1": contact["days_from_test1"],
                        "days_from_test2": contact["days_from_test2"],
                    },
                }
            )

        # Generate spread events (potential transmissions)
        spread_events = []
        for i, contact in enumerate(infection_contacts):
            # Calculate confidence based on temporal proximity
            days_diff = abs(contact["days_from_test1"] - contact["days_from_test2"])
            confidence = max(
                0.1, 1.0 - (days_diff / 28.0)
            )  # Higher confidence for closer test dates

            # Determine direction based on test dates
            p1_test_date = None
            p2_test_date = None

            for _, test in positive_tests.iterrows():
                if test["patient_id"] == contact["patient1"]:
                    p1_test_date = test["collection_date"].strftime("%Y-%m-%d")
                elif test["patient_id"] == contact["patient2"]:
                    p2_test_date = test["collection_date"].strftime("%Y-%m-%d")

            if p1_test_date and p2_test_date:
                # Assume earlier positive test is source
                if p1_test_date <= p2_test_date:
                    source, target = contact["patient1"], contact["patient2"]
                    source_date, target_date = p1_test_date, p2_test_date
                else:
                    source, target = contact["patient2"], contact["patient1"]
                    source_date, target_date = p2_test_date, p1_test_date

                spread_events.append(
                    {
                        "event_id": f"spread_{infection_type}_{i}",
                        "source_patient": source,
                        "target_patient": target,
                        "infection": infection_type,
                        "contact_date": contact["contact_date"].strftime("%Y-%m-%d"),
                        "contact_location": contact["location"],
                        "source_test_date": source_date,
                        "target_test_date": target_date,
                        "days_between_tests": abs(
                            (
                                datetime.strptime(target_date, "%Y-%m-%d")
                                - datetime.strptime(source_date, "%Y-%m-%d")
                            ).days
                        ),
                        "confidence_score": round(confidence, 2),
                    }
                )

        # Generate network nodes
        patient_contact_counts = defaultdict(int)
        for contact in infection_contacts:
            patient_contact_counts[contact["patient1"]] += 1
            patient_contact_counts[contact["patient2"]] += 1

        network_nodes = []
        for patient_id in infection_patients:
            patient_tests = positive_tests[positive_tests["patient_id"] == patient_id]
            first_positive = (
                patient_tests["collection_date"].min()
                if len(patient_tests) > 0
                else None
            )

            # Find cluster membership
            cluster_id = None
            for i, group in enumerate(self.contact_groups):
                if patient_id in group:
                    cluster_id = i + 1
                    break

            network_nodes.append(
                {
                    "id": patient_id,
                    "infections": [infection_type],
                    "primary_infection": infection_type,
                    "first_positive_date": first_positive.strftime("%Y-%m-%d")
                    if first_positive is not None
                    else None,
                    "total_contacts": patient_contact_counts[patient_id],
                    "node_size": min(
                        50, 10 + patient_contact_counts[patient_id] * 5
                    ),  # Visual scaling
                    "cluster_id": cluster_id,
                }
            )

        # Generate network edges
        network_edges = []
        for contact in infection_contacts:
            # Calculate edge strength based on temporal factors
            days_diff = abs(contact["days_from_test1"] - contact["days_from_test2"])
            strength = max(0.1, 1.0 - (days_diff / 28.0))

            network_edges.append(
                {
                    "source": contact["patient1"],
                    "target": contact["patient2"],
                    "infection": infection_type,
                    "contact_date": contact["contact_date"].strftime("%Y-%m-%d"),
                    "location": contact["location"],
                    "strength": round(strength, 2),
                }
            )

        # Sort timeline events by date
        timeline_events.sort(key=lambda x: x["date"])

        # Calculate date range
        if all_dates:
            # Convert all dates to datetime objects for consistent comparison
            datetime_dates = []
            for d in all_dates:
                if hasattr(d, "date"):  # It's already a datetime/timestamp
                    datetime_dates.append(d)
                else:  # It's a date object
                    datetime_dates.append(datetime.combine(d, datetime.min.time()))

            min_date = min(datetime_dates)
            max_date = max(datetime_dates)

            date_range = {
                "earliest": min_date.strftime("%Y-%m-%d"),
                "latest": max_date.strftime("%Y-%m-%d"),
            }
            date_span_days = (max_date.date() - min_date.date()).days
        else:
            date_range = {"earliest": None, "latest": None}
            date_span_days = 0

        # Generate statistics
        stats = {
            "total_patients": len(infection_patients),
            "total_contacts": len(infection_contacts),
            "total_spread_events": len(spread_events),
            "avg_confidence_score": sum(s["confidence_score"] for s in spread_events)
            / len(spread_events)
            if spread_events
            else 0,
            "date_span_days": date_span_days,
            "locations_involved": len(set(c["location"] for c in infection_contacts)),
        }

        return {
            "infection_type": infection_type,
            "timeline_events": timeline_events,
            "spread_events": spread_events,
            "network_nodes": network_nodes,
            "network_edges": network_edges,
            "date_range": date_range,
            "stats": stats,
        }

    def _generate_all_infections_spread(self) -> dict[str, Any]:
        """Generate spread visualization data for all infections."""
        unique_infections = list(set(c["organism"] for c in self.contacts))

        infections_data = {}
        combined_timeline = []
        cross_infection_events = []

        # Generate data for each infection
        for infection in unique_infections:
            infection_data = self._generate_single_infection_spread(infection)
            infections_data[infection] = infection_data
            combined_timeline.extend(infection_data["timeline_events"])

        # Find patients with multiple infections
        patient_infections = defaultdict(set)
        for contact in self.contacts:
            patient_infections[contact["patient1"]].add(contact["organism"])
            patient_infections[contact["patient2"]].add(contact["organism"])

        # Identify cross-infection events
        for patient_id, infections in patient_infections.items():
            if len(infections) > 1:
                patient_tests = self.df_positive[
                    self.df_positive["patient_id"] == patient_id
                ]
                for _, test in patient_tests.iterrows():
                    cross_infection_events.append(
                        {
                            "patient_id": patient_id,
                            "infection": test["infection"],
                            "test_date": test["collection_date"].strftime("%Y-%m-%d"),
                            "all_patient_infections": list(infections),
                        }
                    )

        # Sort combined timeline
        combined_timeline.sort(key=lambda x: x["date"])

        # Global statistics
        global_stats = {
            "total_infections": len(unique_infections),
            "total_patients": len(
                set(p for c in self.contacts for p in [c["patient1"], c["patient2"]])
            ),
            "total_contacts": len(self.contacts),
            "patients_with_multiple_infections": len(
                [p for p, infs in patient_infections.items() if len(infs) > 1]
            ),
            "cross_infection_events": len(cross_infection_events),
            "most_connected_infection": max(
                unique_infections,
                key=lambda x: len([c for c in self.contacts if c["organism"] == x]),
            )
            if unique_infections
            else None,
        }

        return {
            "infections": infections_data,
            "combined_timeline": combined_timeline,
            "cross_infection_events": cross_infection_events,
            "global_stats": global_stats,
        }

    def generate_summary_metrics(
        self, graph_data: dict, clusters: list[dict]
    ) -> dict[str, Any]:
        """Generate summary metrics for the dashboard."""
        total_patients = len(graph_data)
        connected_patients = sum(
            1 for patient in graph_data.values() if patient["contact_count"] > 0
        )
        isolated_patients = total_patients - connected_patients

        # Infection type distribution
        infection_counts = defaultdict(int)
        for patient in graph_data.values():
            for infection in patient["infections"]:
                infection_counts[infection] += 1

        # Contact statistics
        total_contacts = len(self.contacts)

        # Location distribution
        location_counts = defaultdict(int)
        for contact in self.contacts:
            location_counts[contact["location"]] += 1

        return {
            "total_patients": total_patients,
            "connected_patients": connected_patients,
            "isolated_patients": isolated_patients,
            "total_clusters": len(clusters),
            "total_contact_events": total_contacts,
            "infection_distribution": dict(infection_counts),
            "location_distribution": dict(location_counts),
            "largest_cluster_size": max(
                [cluster["patient_count"] for cluster in clusters]
            )
            if clusters
            else 0,
        }

    async def run_detection_pipeline(self) -> dict[str, Any]:
        """Run the complete infection detection pipeline."""
        try:
            logger.info("Starting infection detection pipeline...")

            # Load and process data
            self.load_and_optimize_data()
            self.create_spatial_temporal_index()

            # Detect contacts and clusters
            self.contact_detection()
            self.get_contact_groups()

            # Generate output data
            graph_data = self.generate_graph_data()
            clusters = self.generate_cluster_data()
            summary = self.generate_summary_metrics(graph_data, clusters)

            logger.info("Infection detection pipeline completed successfully")

            return {
                "metadata": {
                    "description": "Infection spreading detection results",
                    "generated_at": datetime.now().isoformat(),
                    "window_days": self.window_days,
                    "data_format_version": "1.0",
                },
                "summary": summary,
                "patients": graph_data,
                "clusters": clusters,
                "contacts": [
                    {
                        "patient1": contact["patient1"],
                        "patient2": contact["patient2"],
                        "location": contact["location"],
                        "organism": contact["organism"],
                        "contact_date": contact["contact_date"].strftime("%Y-%m-%d"),
                        "days_from_test1": contact["days_from_test1"],
                        "days_from_test2": contact["days_from_test2"],
                    }
                    for contact in self.contacts
                ],
            }

        except Exception as e:
            logger.error(f"Error in infection detection pipeline: {str(e)}")
            raise


# Service instance
infection_detection_service = InfectionDetectionService()
