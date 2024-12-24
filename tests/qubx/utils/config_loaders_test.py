import builtins
import os
import pytest
import yaml
import tempfile
from unittest.mock import patch
from pathlib import Path

from qubx.utils.runner import load_strategy_config, StrategyConfig, load_account_env_config


@pytest.fixture
def valid_config_dict():
    """
    Returns a dictionary that matches the expected YAML structure
    for a valid strategy configuration.
    """
    return {
        "config": {
            "strategy": "sty.models.portfolio.pigone.TestPig1",
            "parameters": {
                "top_capitalization_percentile": 2,
                "exchange": "BINANCE.UM",
                "capital_invested": 1000.0,
                "timeframe": "1h",
                "fit_at": "*/30 * * * *",
                "trigger_at": "*/2 * * * *",
                "n_bars_back": 100,
            },
            "exchanges": {
                "BINANCE.UM": {
                    "connector": "ccxt",
                    "universe": ["BTCUSDT", "BNBUSDT"],
                }
            },
            "aux": {
                "reader": "mqdb::nebula"
            },
            "logger": "CsvFileLogsWriter",
            "log_positions_interval": "10Sec",
            "log_portfolio_interval": "5Min"
        }
    }

@pytest.fixture
def mock_env_data():
    """
    Returns a dictionary that represents environment variables,
    including account-specific entries.
    """
    return {
        "BNC-TEST1__APIKEY": "12345",
        "BNC-TEST1__SECRET": "67890",
        "BNC-TEST1__EXCHANGE": "BINANCE.UM",
        "BNC-TEST1__BASE_CURRENCY": "USDT",
        # Some other unrelated env variables
        "ANOTHER_VAR": "not_needed"
    }

class TestLoadStrategyConfig:
    def test_load_strategy_config_valid(
        self,
        valid_config_dict
    ):
        """
        Test that a valid YAML config file is properly loaded and validated.
        """
        valid_yaml = yaml.safe_dump(valid_config_dict)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            tmp_file.write(valid_yaml)
            tmp_file.flush()
            filename = tmp_file.name

        try:
            account_id = "BNC-TEST1"
            strategy_config = load_strategy_config(filename=filename, account=account_id)
        finally:
            Path(filename).unlink(missing_ok=True)

        assert isinstance(strategy_config, StrategyConfig)
        assert strategy_config.strategy == "sty.models.portfolio.pigone.TestPig1"
        assert strategy_config.name == "TestPig1"  # derived from the last part of the path
        assert len(strategy_config.exchanges) == 1
        assert strategy_config.exchanges[0].name == "BINANCE.UM"
        assert strategy_config.exchanges[0].connector == "ccxt"
        assert all([s in strategy_config.exchanges[0].universe for s in ["BTCUSDT", "BNBUSDT"]])
        assert all([s.symbol in ["BTCUSDT", "BNBUSDT"] for s in strategy_config.exchanges[0].instruments])
        assert strategy_config.account == account_id
        assert strategy_config.parameters["timeframe"] == "1h"
        assert strategy_config.logger == "CsvFileLogsWriter"

    def test_load_strategy_config_file_not_found(self):
        """
        Test that if the file doesn't exist, the function raises an exception.
        """
        with pytest.raises(Exception) as exc_info:
            load_strategy_config(filename="non_existing_file.yaml", account="TEST")
        assert "No such file" in str(exc_info.value) or "Can't read strategy config" in str(exc_info.value)

    def test_load_strategy_config_invalid_yaml(self):
        """
        Test that invalid YAML data raises an exception.
        """
        invalid_yaml = ":\ninvalid: ???"

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            tmp_file.write(invalid_yaml)
            tmp_file.flush()
            filename = tmp_file.name

        with pytest.raises(Exception) as exc_info:
            try:
                load_strategy_config(filename=filename, account="TEST")
            finally:
                Path(filename).unlink(missing_ok=True)
        assert "Can't read strategy config" in str(exc_info.value)

    def test_load_strategy_config_missing_keys(self):
        """
        Test that if the required keys are missing, Pydantic validation fails.
        """
        # Missing 'exchanges' key
        config_dict = {
            "config": {
                "strategy": "some.strategy.path",
                # "exchanges": ... is missing
            }
        }
        config_yaml = yaml.safe_dump(config_dict)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            tmp_file.write(config_yaml)
            tmp_file.flush()
            filename = tmp_file.name

        with pytest.raises(Exception) as exc_info:
            try:
                load_strategy_config(filename=filename, account="TEST")
            finally:
                Path(filename).unlink(missing_ok=True)
        assert "exchanges" in str(exc_info.value)


class TestLoadAccountEnvConfig:
    @pytest.mark.parametrize("account_id", ["BNC-TEST1", "bnc-test1", "bnc-TeSt1"])
    def test_load_account_env_config_case_insensitive(
        self,
        account_id,
        mock_env_data
    ):
        """
        Test that the loader will retrieve environment variables.
        """
        with patch.dict(os.environ, mock_env_data, clear=True):
            with patch("dotenv.find_dotenv", return_value=""):
                account_data = load_account_env_config(account_id=account_id, env_file=".env")

        assert account_data is not None
        assert "apiKey".lower() in account_data
        assert account_data["apikey"] == "12345"
        assert account_data["secret"] == "67890"
        assert account_data["exchange"] == "BINANCE.UM"
        assert account_data["base_currency"] == "USDT"
        assert account_data["account_id"].lower() == account_id.lower()

    def test_load_account_env_config_no_records_found(self):
        """
        Test when the environment does not have any variables matching the account ID.
        Expect a logger error and an empty or None result.
        """
        mock_env_data = {
            "SOME_OTHER__APIKEY": "12345"
        }

        with patch.dict(os.environ, mock_env_data, clear=True):
            with patch("dotenv.find_dotenv", return_value=""):
                with patch("qubx.logger.error") as mock_logger_error:
                    account_data = load_account_env_config(
                        account_id="BNC-TEST1",
                        env_file=".env"
                    )

        assert account_data is None
        mock_logger_error.assert_called_once()
        assert "Can't find exchange for" in str(mock_logger_error.call_args[0][0])
