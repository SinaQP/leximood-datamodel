from pathlib import Path

import pandas as pd
import pytest

from leximood.train import TrainConfig, load_dataset


def test_train_config_defaults_are_stable():
    config = TrainConfig()
    assert config.vocab_size == 15000
    assert config.max_len == 100
    assert config.epochs == 15


def test_load_dataset_requires_expected_columns(tmp_path: Path):
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"message": ["hello"], "label": ["joy"]}).to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="Missing required column"):
        load_dataset(bad_path)
