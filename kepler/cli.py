"""
CLI entry point for Kepler framework

This module provides the main entry point for the CLI as defined in pyproject.toml
"""

from kepler.cli.main import app

if __name__ == "__main__":
    app()