from pathlib import Path

import click

from qubx.utils.misc import add_project_to_system_path, logo
from qubx.utils.runner.runner import run_strategy_yaml, run_strategy_yaml_in_jupyter, simulate_strategy


@click.group()
def main():
    """
    Qubx CLI.
    """
    pass


@main.command()
@click.argument("config-file", type=Path, required=True)
@click.option(
    "--account-file",
    "-a",
    type=Path,
    help="Account configuration file path.",
    required=False,
)
@click.option("--paper", "-p", is_flag=True, default=False, help="Use paper trading mode.", show_default=True)
@click.option(
    "--jupyter", "-j", is_flag=True, default=False, help="Run strategy in jupyter console.", show_default=True
)
def run(config_file: Path, account_file: Path | None, paper: bool, jupyter: bool):
    """
    Starts the strategy with the given configuration file. If paper mode is enabled, account is not required.

    Account configurations are searched in the following priority:\n
    - If provided, the account file is searched first.\n
    - If exists, accounts.toml located in the same folder with the config searched.\n
    - If neither of the above are provided, the accounts.toml in the ~/qubx/accounts.toml path is searched.
    """
    add_project_to_system_path()
    add_project_to_system_path(str(config_file.parent))
    if jupyter:
        run_strategy_yaml_in_jupyter(config_file, account_file, paper)
    else:
        logo()
        run_strategy_yaml(config_file, account_file, paper, blocking=True)


@main.command()
@click.argument("config-file", type=Path, required=True)
@click.option(
    "--start", "-s", default=None, type=str, help="Override simulation start date from config.", show_default=True
)
@click.option(
    "--end", "-e", default=None, type=str, help="Override simulation end date from config.", show_default=True
)
@click.option(
    "--output", "-o", default="results", type=str, help="Output directory for simulation results.", show_default=True
)
def simulate(config_file: Path, start: str | None, end: str | None, output: str | None):
    add_project_to_system_path()
    add_project_to_system_path(str(config_file.parent))
    logo()
    simulate_strategy(config_file, output, start, end)


if __name__ == "__main__":
    main()
