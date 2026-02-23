#!/bin/bash

ARCHIVE_FILE="files-archive"
EXTRACT_DIR="SAR"

echo "Downloading SAR DATA from Zenodo..."
wget -v https://zenodo.org/api/records/17397954/files-archive

if [ ! -f "$ARCHIVE_FILE" ]; then
    echo "Error: Failed to download archive"
    exit 1
fi

echo "Creating extraction directory: $EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"

echo "Extracting archive..."
if file "$ARCHIVE_FILE" | grep -q "Zip"; then
    echo "Detected ZIP archive, extracting..."
    unzip -q "$ARCHIVE_FILE" -d "$EXTRACT_DIR"
elif file "$ARCHIVE_FILE" | grep -q "tar"; then
    echo "Detected TAR archive, extracting..."
    tar -xf "$ARCHIVE_FILE" -C "$EXTRACT_DIR"
else
    echo "Attempting generic extraction..."
    unzip -q "$ARCHIVE_FILE" -d "$EXTRACT_DIR" || tar -xf "$ARCHIVE_FILE" -C "$EXTRACT_DIR"
fi

if [ $? -eq 0 ]; then
    echo "Extraction successful"
    echo "Removing archive: $ARCHIVE_FILE"
    rm "$ARCHIVE_FILE"
    echo "Done. Data extracted to: $EXTRACT_DIR"
else
    echo "Error: Extraction failed"
    exit 1
fi
