#!/bin/bash

set -e

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# print all functions in this file
function help {
    echo "$0 <task> <args>"
    echo "Tasks:"
    compgen -A function | cat -n
}

# remove all files generated by tests, builds, or operating this codebase
function clean {
    rm -rf dist build coverage.xml test-reports out
    find . \
      -type d \
      \( \
        -name "*cache*" \
        -o -name "*.dist-info" \
        -o -name "*.egg-info" \
        -o -name "*htmlcov" \
      \) \
      -not -path "*env/*" \
      -exec rm -r {} + || true

    find . \
      -type f \
      -name "*.pyc" \
      -not -path "*env/*" \
      -exec rm {} +

    # Remove log files
    LOG_DIR="${THIS_DIR}/logs"
    if [ -d "$LOG_DIR" ]; then
        rm -rf "$LOG_DIR"
    fi

    TEST_LOG_DIR="${THIS_DIR}/logs_test"
    if [ -d "$TEST_LOG_DIR" ]; then
        rm -rf "$TEST_LOG_DIR"
    fi

    TEST_LOG_DIR="${THIS_DIR}/control"
    if [ -d "$TEST_LOG_DIR" ]; then
        rm -rf "$TEST_LOG_DIR"
    fi
}

function install:package {
    uv pip install -e .
}


# install core and development Python dependencies into the currently activated venv
function install:dev {
    uv sync --group dev
}

function install:docs {
    uv sync --group docs
}

function model {
    python -m nextpredco.core.model
}

function pre-commit:install {
    pre-commit install
}

function pre-commit:run {
    pre-commit run --all-files
}

function pre-commit:update {
    pre-commit autoupdate
}

function test {
    pytest ./tests --verbose
}

function test:cov {
    pytest --cov=nextpredco --cov-report=xml --cov-report=term-missing ./tests
}
function test:model {
    python tests/units/test_model.py
}

TIMEFORMAT="Task completed in %3lR"
time ${@:-help}
