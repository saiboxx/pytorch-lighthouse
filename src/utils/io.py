"""Classes and functions for handling IO operations."""

import yaml  # type: ignore


def read_yml(filepath: str) -> dict:
    """Load a yml file to memory as dict."""
    with open(filepath, 'r') as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))
