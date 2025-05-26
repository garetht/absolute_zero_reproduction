#!/bin/bash
pip install uv
uv venv
source .venv/bin/activate
uv sync
