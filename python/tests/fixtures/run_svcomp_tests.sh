#!/bin/bash
# Run all SV-COMP style fixtures through verith + lake build
# Must be run from the repo root: bash python/tests/fixtures/run_svcomp_tests.sh

set -e
VERITH=".venv/bin/verith"
PASS=0
FAIL=0

run_test() {
    local name=$1; shift
    echo "=== $name ==="
    local outdir="/tmp/svcomp_$name"
    rm -rf "$outdir"
    $VERITH "$@" -o "$outdir" 2>&1 | tail -1
    cd "$outdir/Rea"
    lake update 2>&1 | tail -1
    local result=$(lake build Certificate 2>&1)
    local sorry_count=$(echo "$result" | grep "warning.*Certificate.*sorry" | wc -l)
    local error_count=$(echo "$result" | grep "^error:" | wc -l)
    if [ "$error_count" -gt 0 ]; then
        echo "  FAIL: $error_count error(s)"
        echo "$result" | grep "error:" | head -3
        FAIL=$((FAIL + 1))
    elif [ "$sorry_count" -gt 0 ]; then
        echo "  FAIL: $sorry_count sorry warning(s)"
        echo "$result" | grep "sorry" | grep Certificate | head -3
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: zero sorry, zero errors"
        PASS=$((PASS + 1))
    fi
    cd /Users/zehariel/Documents/Code/reactive-modules
    echo ""
}

echo "Running SV-COMP style termination tests..."
echo ""

# 1. Simple countdown
run_test "countdown" \
    python/tests/fixtures/svcomp_countdown.py \
    --property "(= s0 0)" \
    --invariant "(and (>= s0 0) (<= s0 100))" \
    --ranking "(ite (= s0 0) 0 s0)"

# 2. Two variables
run_test "twovars" \
    python/tests/fixtures/svcomp_twovars.py \
    --property "(= s0 s1)" \
    --invariant "(and (>= s0 0) (<= s0 s1) (= s1 10))" \
    --ranking "(ite (= s0 s1) 0 (- s1 s0))"

# 3. Nested loops (harder)
run_test "nested" \
    python/tests/fixtures/svcomp_nested.py \
    --property "(and (= s0 0) (= s1 0))" \
    --invariant "(and (>= s0 0) (<= s0 3) (>= s1 0) (<= s1 3))" \
    --ranking "(ite (and (= s0 0) (= s1 0)) 0 (+ (* (- 3 s0) 4) (- 3 s1)))"

# 4. Bounded Collatz
run_test "collatz" \
    python/tests/fixtures/svcomp_collatz_bounded.py \
    --property "(= s0 1)" \
    --invariant "(and (>= s0 1) (<= s0 8))" \
    --ranking "(ite (= s0 1) 0 (- s0 1))"

# 5. GCD
run_test "gcd" \
    python/tests/fixtures/svcomp_gcd.py \
    --property "(= s0 s1)" \
    --invariant "(and (>= s0 1) (>= s1 1))" \
    --ranking "(ite (= s0 s1) 0 (+ (- s0 1) (- s1 1)))"

# 6. Original counter (regression)
run_test "counter" \
    python/tests/fixtures/counter.py \
    --property "(= s0 0)" \
    --invariant "(and (>= s0 0) (<= s0 9))" \
    --ranking "(ite (= s0 0) 0 (- 10 s0))"

# 7. Original twobit (regression)
run_test "twobit" \
    python/tests/fixtures/twobit.py \
    --property "(and (= s0 false) (= s1 false))" \
    --pre "(= e0 true)" \
    --invariant "true" \
    --ranking "(ite s0 (ite s1 1 3) (ite s1 2 0))"

echo "================================"
echo "Results: $PASS passed, $FAIL failed"
echo "================================"
