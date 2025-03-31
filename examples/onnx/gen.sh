#!/bin/bash

# Script to run gen.py in all subdirectories where it exists

# Find all directories containing gen.py
dirs=$(find . -name "gen.py" -type f | sort | xargs dirname)

# Loop through each directory and run gen.py
for dir in $dirs; do
  echo "Running gen.py in $dir"
  cd "$dir" || continue
  python3 gen.py
  cd - > /dev/null  # Return to original directory silently
done

echo "Completed all gen.py executions"