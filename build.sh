#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ----------------------------
# Config
# ----------------------------
PACKAGE_NAME="xentropy"
DIST_DIR="dist"
USE_TESTPYPI=false   # set to true to upload to TestPyPI

# PyPI repository URLs
PYPI_URL="https://upload.pypi.org/legacy/"
TESTPYPI_URL="https://test.pypi.org/legacy/"

# ----------------------------
# Clean previous builds
# ----------------------------
echo "Cleaning previous builds..."
rm -rf ${DIST_DIR} *.egg-info

# ----------------------------
# Build the package
# ----------------------------
echo "Building the package..."
python -m build

# ----------------------------
# Upload to PyPI / TestPyPI
# ----------------------------
if [ "$USE_TESTPYPI" = true ]; then
    echo "Uploading to TestPyPI..."
    python -m twine upload --repository-url $TESTPYPI_URL ${DIST_DIR}/*
else
    echo "Uploading to PyPI..."
    python -m twine upload --repository-url $PYPI_URL ${DIST_DIR}/*
fi

echo "✅ Package $PACKAGE_NAME uploaded successfully."


