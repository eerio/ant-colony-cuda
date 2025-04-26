#!/usr/bin/env python3

import os
import re
import sys

def extract_instance_size(filename):
    """Extracts the last number before '_worker.out'."""
    base = os.path.basename(filename).replace('_worker.out', '')
    numbers = re.findall(r'\d+', base)
    if not numbers:
        raise ValueError(f"No number found in filename {filename}")
    return int(numbers[-1])

def read_cost(filepath):
    """Reads the first two lines: cost and tour."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if len(lines) < 1:
        raise ValueError(f"File {filepath} does not have at least one line")
    cost = float(lines[0].strip())
    return cost

def read_cost_and_tour(filepath):
    """Reads the first two lines: cost and tour."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"File {filepath} does not have at least two lines")
    cost = float(lines[0].strip())
    tour = list(map(int, lines[1].strip().split()))
    return cost, tour

def main():
    ok = True
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('_worker.out'):
                worker_path = os.path.join(root, file)
                tsplib_path = worker_path.replace('_worker.out', '_tsplibsolution.out')

                print(f"Checking {worker_path}...")

                if not os.path.exists(tsplib_path):
                    print(f"  ERROR: Missing TSPLIB file {tsplib_path}")
                    ok = False
                    sys.exit(1)
                    continue

                try:
                    instance_size = extract_instance_size(file)
                    worker_cost, worker_tour = read_cost_and_tour(worker_path)
                    tsplib_cost = read_cost(tsplib_path)

                    # Check TSPLIB cost sanity
                    if tsplib_cost <= 10:
                        print(f"  ERROR: TSPLIB cost too small ({tsplib_cost})")
                        ok = False
                        sys.exit(1)
                        continue

                    # Cost comparisons
                    if worker_cost < tsplib_cost:
                        print(f"  ERROR: Worker cost {worker_cost} smaller than TSPLIB cost {tsplib_cost}")
                        ok = False
                        sys.exit(1)

                    k = 2
                    if worker_cost > k * tsplib_cost:
                        print(f"  ERROR: Worker cost {worker_cost} more than {k}x TSPLIB cost {tsplib_cost}")
                        ok = False
                        sys.exit(1)

                    # Tour validity check
                    expected_tour = set(range(1, instance_size + 1))
                    if set(worker_tour) != expected_tour:
                        print(f"  ERROR: Tour is not a valid permutation of 1..{instance_size}")
                        print(f"    Got: {sorted(worker_tour)}")
                        print(f"    Expected: {list(expected_tour)}")
                        ok = False
                        sys.exit(1)

                except Exception as e:
                    print(f"  ERROR: {e}")
                    ok = False
                    sys.exit(1)

                if ok:
                    print(f"  OK: {worker_path}")

    if not ok:
        sys.exit(1)

if __name__ == '__main__':
    main()
