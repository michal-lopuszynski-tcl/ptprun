import logging

import pytest

import ptprun.extraction

import sample_dags  # isort:skip

logger = logging.getLogger(__name__)


def _match_orbits(true_orbits, found_orbits, verbose=True):
    matched_orbits = []
    for orbit in true_orbits:
        for found_orbit in found_orbits:
            if found_orbit == orbit:
                matched_orbits += [orbit]

    unmatched_true_orbits = [o for o in true_orbits if o not in matched_orbits]
    unmatched_found_orbits = [o for o in found_orbits if o not in matched_orbits]
    return matched_orbits, unmatched_true_orbits, unmatched_found_orbits


TEST_ORBITS_DATA = [[k, *v()] for k, v in sample_dags.DAG_DATA_FACTORIES.items()]

TEST_EXTENDED_ORBITS_DATA = [[v[0], v[1], v[2]] for v in TEST_ORBITS_DATA]

TEST_FINAL_ORBITS_DATA = [[v[0], v[1], v[3]] for v in TEST_ORBITS_DATA]


@pytest.mark.parametrize(
    "name, graph_module, true_extended_orbits", TEST_EXTENDED_ORBITS_DATA
)
def test_extended_orbits(name, graph_module, true_extended_orbits):
    found_extended_orbits = ptprun.extraction._extract_extended_orbits(graph_module)

    (
        matched_exteded_orbits,
        unmatched_true_extended_orbits,
        unmatched_found_extended_orbits,
    ) = _match_orbits(
        true_orbits=true_extended_orbits,
        found_orbits=found_extended_orbits,
    )
    if unmatched_true_extended_orbits or unmatched_found_extended_orbits:
        logger.warning(f"{name} - {len(matched_exteded_orbits)} correctly found orbits")
        for i, o in enumerate(matched_exteded_orbits, start=1):
            logger.warning(f"{name} - {i}. {o}")

    if unmatched_found_extended_orbits:
        msg = f"{name} - {name} - {len(unmatched_found_extended_orbits)}"
        msg += " incorrectly found orbits"
        logger.warning(msg)
        for i, o in enumerate(unmatched_found_extended_orbits, start=1):
            logger.warning(f"{name} - {i}. {o}")

    if unmatched_true_extended_orbits:
        msg = f"{name} - {name} - {len(unmatched_true_extended_orbits)}"
        msg += " missed orbits"
        logger.warning(msg)
        for i, o in enumerate(unmatched_true_extended_orbits, start=1):
            logger.warning(f"{name} - {i}. {o}")

    assert len(matched_exteded_orbits) == len(true_extended_orbits)
    assert len(unmatched_true_extended_orbits) == 0
    assert len(unmatched_found_extended_orbits) == 0


@pytest.mark.parametrize(
    "name, graph_module, true_final_orbits", TEST_FINAL_ORBITS_DATA
)
def test_final_orbits(name, graph_module, true_final_orbits):
    found_final_orbits = ptprun.extraction.extract_orbits(graph_module)
    (
        matched_final_orbits,
        unmatched_true_final_orbits,
        unmatched_found_final_orbits,
    ) = _match_orbits(
        true_orbits=true_final_orbits,
        found_orbits=found_final_orbits,
    )
    assert len(matched_final_orbits) == len(true_final_orbits)
    assert len(unmatched_true_final_orbits) == 0
    assert len(unmatched_found_final_orbits) == 0
