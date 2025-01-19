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
    rm -rf dist build coverage.xml test-reports
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
    
    TEST_LOG_DIR="${LOG_DIR}/logs_test"
    if [ -d "$TEST_LOG_DIR" ]; then
        rm -rf "$TEST_LOG_DIR"
    fi
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

TIMEFORMAT="Task completed in %3lR"
time ${@:-help}
