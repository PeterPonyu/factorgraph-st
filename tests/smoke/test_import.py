def test_import_top_level():
    import factorgraph_st

    assert hasattr(factorgraph_st, "__version__")
    assert isinstance(factorgraph_st.__version__, str)
    assert factorgraph_st.__version__ != ""


def test_no_module_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    import factorgraph_st as m

    importlib.reload(m)
    assert m.__version__ != ""
