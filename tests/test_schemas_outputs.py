import numpy as np
import pytest

from factorgraph_st.schemas import SchemaError, validate_outputs


def _ok():
    n_spots, n_genes, K_s, K_p, d = 6, 8, 2, 1, 3
    return dict(
        H=np.zeros((n_spots, d), dtype=np.float32),
        W=np.ones((n_genes, K_s + K_p), dtype=np.float32),
        Z_shared=np.ones((n_spots, K_s), dtype=np.float32),
        Z_private=np.zeros((n_spots, K_p), dtype=np.float32),
        domain_id=np.zeros(n_spots, dtype=np.int64),
        n_spots=n_spots,
        n_genes=n_genes,
    )


def test_valid_outputs_pass():
    validate_outputs(**_ok())


def test_W_negative_fails():
    kw = _ok()
    kw["W"] = kw["W"].copy()
    kw["W"][0, 0] = -1.0
    with pytest.raises(SchemaError, match="W must be nonnegative"):
        validate_outputs(**kw)


def test_Z_shared_negative_fails():
    kw = _ok()
    kw["Z_shared"] = kw["Z_shared"].copy()
    kw["Z_shared"][0, 0] = -0.5
    with pytest.raises(SchemaError, match="Z_shared must be nonnegative"):
        validate_outputs(**kw)


def test_Z_private_negative_fails():
    kw = _ok()
    kw["Z_private"] = kw["Z_private"].copy()
    kw["Z_private"][0, 0] = -1.0
    with pytest.raises(SchemaError, match="Z_private must be nonnegative"):
        validate_outputs(**kw)


def test_W_column_K_mismatch():
    kw = _ok()
    kw["W"] = np.ones((kw["n_genes"], 999), dtype=np.float32)
    with pytest.raises(SchemaError, match="K_shared \\+ K_private"):
        validate_outputs(**kw)


def test_H_wrong_n_spots():
    kw = _ok()
    kw["H"] = np.zeros((kw["n_spots"] - 1, 3), dtype=np.float32)
    with pytest.raises(SchemaError, match="H must be"):
        validate_outputs(**kw)


def test_domain_id_wrong_dtype():
    kw = _ok()
    kw["domain_id"] = np.zeros(kw["n_spots"], dtype=np.float32)
    with pytest.raises(SchemaError, match="domain_id must be integer"):
        validate_outputs(**kw)


def test_domain_id_negative_fails():
    kw = _ok()
    kw["domain_id"] = np.array([-1, 0, 0, 0, 0, 0], dtype=np.int64)
    with pytest.raises(SchemaError, match="non-negative"):
        validate_outputs(**kw)
