# -*- coding: utf-8 -*-
"""Smoke tests to verify all package imports work."""


def test_core_imports():
    from dwipada.core import analyze_dwipada, DWIPADA_RULES_BLOCK
    from dwipada.core.aksharanusarika import split_aksharalu, akshara_ganavibhajana
    assert callable(analyze_dwipada)
    assert isinstance(DWIPADA_RULES_BLOCK, str)
    assert callable(split_aksharalu)


def test_paths_imports():
    from dwipada.paths import PROJECT_ROOT, DATA_DIR, CONFIG_FILE
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.name == "data"


def test_data_base_imports():
    from dwipada.data.clean_base import clean_line, CHARS_TO_REMOVE
    assert callable(clean_line)
    assert len(CHARS_TO_REMOVE) == 25


def test_dataset_imports():
    from dwipada.dataset import stats, create, augment, combine, prepare_synthetic
    assert hasattr(stats, 'main')


def test_batch_config_import():
    from dwipada.batch.config import load_api_key, load_vertex_config
    assert callable(load_api_key)


def test_version():
    import dwipada
    assert dwipada.__version__ == "1.0.0"
