import click
from .index import INDEX
from .runner import Runner
from .nets.net import UnsupportedNetParamError
from typing import Dict, Any

@click.group()
def cli():
    pass

@cli.group()
def show():
    """
    Show information abour register components
    """
    pass

@show.command()
def backends():
    for name, b in INDEX['backends'].items():
        print("")
        print(f"name: {name}")
        print(f"description: {b.desc}")
        print(f"shape: {b.shape}")

@show.command()
def langs():
    for name, l in INDEX['langs'].items():
        print("")
        print(f"name: {name}")
        print(f"description: {l.desc}")
        print(f"shape: {l.shape}")
        print(f"alphabet size: {l.alphabet_size}")
        print(f"extra params: {l.extra_params}")

@show.command()
def nets():
    for name, n in INDEX['nets'].items():
        print("")
        print(f"name: {name}")
        print(f"description: {n.desc}")
        print(f"extra params: {n.extra_params}")

@cli.command()
@click.argument('config')
def run(config: str):
    """
    Run a given config
    """
    runner = Runner(config)
    try:
        runner.run()
    except UnsupportedNetParamError as e:
        print(f"[!] Error - unsupported parameter: {e}")
