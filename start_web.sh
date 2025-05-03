#!/bin/bash

# Step 1: Remove old database
echo "Removing old database..."
rm -f test.db

# Step 2: Remove uploads and summaries
echo "Cleaning up old uploads and summaries..."
rm -rf uploads summaries

# Step 3: Recreate directories
mkdir -p uploads summaries

# Step 4: Initialize database
echo "Initializing database..."
python3 -m web.db.init_db

export PYDANTIC_V2_FORCE_V1_BEHAVIOR=true

# Step 5: Start development server
echo "Starting development server..."
inv dev
