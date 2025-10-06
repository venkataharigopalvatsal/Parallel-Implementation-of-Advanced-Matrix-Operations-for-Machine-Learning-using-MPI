#!/bin/bash
# File: test_automation.sh

# --- Configuration ---
# Name of your parallel MPI/OpenMP executable
PARALLEL_EXEC="hybrid_summa"
# Name of your main C source file
PARALLEL_SRC="hpc.c" #<-- Assuming your file is hpc.c

# Name of the serial verifier executable and source
SERIAL_EXEC="serial_verifier"
SERIAL_SRC="serial_verifier.c"

# Test configurations
SIZES="64 128 256 512"
PROCESSES="1 2 4 8"
VERIFY_SIZE=8

# Output file for performance data
RESULTS_FILE="performance_results.csv"


# --- Script Body ---
echo "--- Automated Test and Performance Script ---"

# 1. COMPILE CODES
echo "Step 1: Compiling source files..."
mpicc -fopenmp -o $PARALLEL_EXEC $PARALLEL_SRC -lm
if [ $? -ne 0 ]; then
    echo "ERROR: Parallel code compilation failed. Aborting."
    exit 1
fi

gcc -o $SERIAL_EXEC $SERIAL_SRC
if [ $? -ne 0 ]; then
    echo "ERROR: Serial verifier compilation failed. Aborting."
    exit 1
fi
echo "Compilation successful."
echo ""


# 2. VERIFY CORRECTNESS (for a small matrix)
echo "Step 2: Verifying correctness for a small ${VERIFY_SIZE}x${VERIFY_SIZE} matrix..."
./$SERIAL_EXEC $VERIFY_SIZE m > serial_output.txt

# The "2>/dev/null" part is CRITICAL: it discards stderr.
mpirun -np 1 ./$PARALLEL_EXEC $VERIFY_SIZE m > parallel_output.txt 2>/dev/null

# Compare the results
diff -q serial_output.txt parallel_output.txt
if [ $? -eq 0 ]; then
    echo "✅ CORRECTNESS CHECK: PASSED"
else
    echo "❌ CORRECTNESS CHECK: FAILED. Output differs from serial version."
    echo "    Check serial_output.txt and parallel_output.txt for differences."
fi
rm serial_output.txt parallel_output.txt
echo ""


# 3. RUN PERFORMANCE TESTS
echo "Step 3: Running performance tests..."
echo "MatrixSize,Processes,TimeElapsed" > $RESULTS_FILE

for N in $SIZES; do
    for P in $PROCESSES; do
        if [ $P -ne 1 ] && [ $P -ne 4 ] && [ $P -ne 9 ] && [ $P -ne 16 ]; then
             echo "Skipping test for P=${P}: Cannon's algorithm requires a square grid (1, 4, 9, 16... processes)."
             continue
        fi

        echo "Running test: N=${N}, Processes=${P}"
        
        # The "2>&1" part merges stderr with stdout so we can grep it
        output=$(mpirun -np $P ./$PARALLEL_EXEC $N m 2>&1)
        
        # CORRECTED: Use awk '{print $3}' to get the 3rd word
        time_elapsed=$(echo "$output" | grep "Time Elapsed" | awk '{print $3}')
        
        if [ -z "$time_elapsed" ]; then
            echo "  WARNING: Could not extract time for N=${N}, P=${P}. Check for errors."
        else
            echo "${N},${P},${time_elapsed}" >> $RESULTS_FILE
            echo "  Time: ${time_elapsed}s. Result saved."
        fi
    done
done

echo ""
echo "--- All tests complete. ---"
echo "Performance data saved in ${SESSION_RESULTS_FILE}"
