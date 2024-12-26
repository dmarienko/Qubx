from pathlib import Path

import click

from qubx.utils.misc import add_project_to_system_path, logo
from qubx.utils.runner.runner import run_strategy_yaml


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
    logo()
    add_project_to_system_path()
    add_project_to_system_path(str(config_file.parent))
    run_strategy_yaml(config_file, account_file, paper, jupyter, blocking=True)


if __name__ == "__main__":
    main()
