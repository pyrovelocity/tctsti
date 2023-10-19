import click

from tctsti import __version__


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="tctsti")
def main():
    click.echo("tctsti")
