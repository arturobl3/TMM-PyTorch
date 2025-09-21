#!/usr/bin/env python3
"""Test that epsilon method also enforces bounds checking."""

import torch
import sys
import os

# Add the TMM-PyTorch root directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_tmm.dispersion import TabulatedData


def test_epsilon_bounds():
    """Test that epsilon method also enforces bounds checking."""
    print("Testing epsilon method bounds checking...")

    # Create sample data
    wavelengths = torch.linspace(500, 1000, 6)
    n_values = torch.linspace(1.4, 1.6, 6) + 0.01j
    dispersion = TabulatedData(wavelengths, n_values)

    # Test within bounds (should work)
    try:
        eps = dispersion.epsilon(torch.tensor([750.0]))
        print(f"✓ epsilon(750 nm) = {eps[0]:.4f}")
    except ValueError as e:
        print(f"✗ Unexpected error: {e}")

    # Test outside bounds (should raise error)
    try:
        eps = dispersion.epsilon(torch.tensor([400.0]))
        print(f"✗ Should have failed for 400 nm")
    except ValueError as e:
        print(f"✓ epsilon method correctly raises error: {e}")


if __name__ == "__main__":
    test_epsilon_bounds()
