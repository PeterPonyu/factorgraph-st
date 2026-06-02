import numpy as np

from factorgraph_st.eval.metrics import adjusted_rand_index, morans_i
from factorgraph_st.synth.generator import _assign_spatial_domains, generate_instance


def test_adjusted_rand_index_perfect_and_randomish_cases():
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    assert adjusted_rand_index(labels, labels.copy()) == 1.0
    shuffled = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    assert adjusted_rand_index(labels, shuffled) < 0.2


def test_morans_i_range_on_generator_domains():
    inst = generate_instance(n_sections=2, n_spots_per_section=10, n_domains=3, seed=13)
    value = morans_i(inst.domain_id, inst.edges)
    assert -1.0 <= value <= 1.0


def test_spatial_domains_are_section_aware():
    """Two sections with identical coordinates must receive section-specific
    domain partitions, while the label vocabulary stays bounded by n_domains (#74)."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.0, 1.0, size=(12, 2)).astype(np.float32)
    coords = np.concatenate([base, base], axis=0)
    section_id = np.array([0] * 12 + [1] * 12, dtype=np.int64)

    labels = _assign_spatial_domains(coords, section_id, n_domains=3, seed=5)
    sec0, sec1 = labels[:12], labels[12:]

    # Section-aware: identical geometry, different partition per section.
    assert not np.array_equal(sec0, sec1)
    # Label vocabulary bounded by n_domains (shared across sections).
    assert np.unique(labels).size <= 3
    # Each section still uses the full label set when it has enough spots.
    assert np.unique(sec0).size == 3 and np.unique(sec1).size == 3
    # Deterministic in (seed, section index).
    np.testing.assert_array_equal(
        labels, _assign_spatial_domains(coords, section_id, n_domains=3, seed=5)
    )
