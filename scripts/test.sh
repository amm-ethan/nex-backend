#!/usr/bin/env bash

set -e
set -x

PYTHONPATH=. coverage run --source=app -m pytest
PYTHONPATH=. coverage report --show-missing
PYTHONPATH=. coverage html --title "${@-coverage}"
