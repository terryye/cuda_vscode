#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Function to run tests for scripts matching a pattern in subfolders
run_tests() {
    pattern=$1

    #if pattern is run_modal_*.sh, check if modal is installed
    if [[ $pattern == "run_modal_*.sh" ]]; then
        if ! command -v modal &> /dev/null; then   
            echo -e "\033[31m modal is not installed, skipping modal tests. \033[0m"
            return
        fi
    fi

    # Script to test all run_modal_* scripts in subfolders
    # For each subfolder, run the run_modal_*.sh script and check if "Test Pass" is in the output
    for dir in */; do
        if [ -d "$dir" ]; then
            echo "Testing running $dir"
            cd "$dir"
            # Find the script matching the pattern
            script=$(ls $pattern 2>/dev/null | head -1)
            if [ -n "$script" ]; then
                echo "Running: ./$script, this may take a couple of minutes."
                # Run the script and capture output
                output=$(./$script 2>&1)
                if echo "$output" | grep -q "Test Pass"; then
                    echo -e "$dir: \033[32m pass \033[0m"
                else
                    echo -e "$dir: \033[31m wrong \033[0m"
                    echo "Output was:"
                    echo "$output"
                    echo "---"
                fi
            else
                echo "$dir: no script matching $pattern found"
            fi
            cd ..
        fi
    done
}

#Run the tests for run_modal_*.sh scripts
echo "Starting tests for modal execution scripts..."
run_tests "run_modal_*.sh"

echo "Starting tests for local execution scripts..."
run_tests "run_local_*.sh"