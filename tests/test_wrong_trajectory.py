"""TDD tests for wrong-trajectory substitution experiment."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.network import RecurrentMLP
from src.training import generate_data, train


def test_wrong_trajectory_matching():
    """Verify trial matching logic finds valid matches."""
    # Train a small model
    net = RecurrentMLP(seed=0)
    X, y = generate_data(50, noise_level=0.5, seed=0)
    train(net, X, y, epochs=100, lr=0.01, time_weights=[0.0, 0.2, 1.0])

    # Get t=1 predictions for all trials
    preds_t1 = []
    true_labels = []
    for i in range(len(X)):
        outputs, _ = net.forward_sequence(X[i], T=3)
        preds_t1.append(np.argmax(outputs[0]))
        true_labels.append(np.argmax(y[i]))

    preds_t1 = np.array(preds_t1)
    true_labels = np.array(true_labels)
    correct_t1 = (preds_t1 == true_labels)

    # Build match index: group by (predicted_class, correctness)
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(len(X)):
        key = (preds_t1[i], correct_t1[i])
        groups[key].append(i)

    # For each trial, find a match
    matched = 0
    total = 0
    for i in range(len(X)):
        key = (preds_t1[i], correct_t1[i])
        candidates = [j for j in groups[key] if j != i]
        if candidates:
            matched += 1
            # Verify match properties
            j = candidates[0]
            assert preds_t1[j] == preds_t1[i], "Matched trial must have same predicted class"
            assert correct_t1[j] == correct_t1[i], "Matched trial must have same correctness"
            assert j != i, "Matched trial must be different"
        total += 1

    # At least some trials should have matches
    assert matched > 0, "Should find at least some matching trials"
    print(f"Matched {matched}/{total} trials")


def test_wrong_trajectory_feedback_injection():
    """Verify that injecting wrong-trial feedback produces different output than self."""
    net = RecurrentMLP(seed=0)
    X, y = generate_data(20, noise_level=0.5, seed=0)
    train(net, X, y, epochs=200, lr=0.01, time_weights=[0.0, 0.2, 1.0])

    # Normal forward
    outputs_self, _ = net.forward_sequence(X[0], T=3)

    # Wrong-trial forward: inject output from trial 1 as feedback at t=2
    net.reset_state()
    _ = net.forward(X[0])  # t=1

    # Get trial 1's t=1 output
    outputs_other, _ = net.forward_sequence(X[1], T=3)

    # Inject other trial's t=1 output
    net._prev_output = outputs_other[0].copy()
    net._has_feedback = True
    y_t2_wrong = net.forward(X[0])

    # Outputs should differ
    assert not np.allclose(outputs_self[1], y_t2_wrong), \
        "Wrong-trial feedback should produce different t=2 output"
