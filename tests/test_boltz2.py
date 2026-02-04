"""Tests for Boltz-2 predictor."""

import pytest
import numpy as np
from src.boltz2 import Boltz2Predictor, AffinityCalculator


def test_delta_g_to_kd():
    """Test ΔG to Kd conversion."""
    # ΔG = -10 kcal/mol should give low nM Kd
    delta_g = -10.0
    kd = AffinityCalculator.delta_g_to_kd(delta_g)
    
    assert kd > 0
    assert kd < 100  # Should be in low nM range


def test_kd_to_delta_g():
    """Test Kd to ΔG conversion."""
    # 10 nM Kd
    kd_nm = 10.0
    delta_g = AffinityCalculator.kd_to_delta_g(kd_nm)
    
    assert delta_g < 0  # Favorable binding
    assert -12 < delta_g < -8  # Reasonable range


def test_classify_affinity():
    """Test affinity classification."""
    assert "Very Strong" in AffinityCalculator.classify_affinity(0.5)
    assert "Strong" in AffinityCalculator.classify_affinity(5.0)
    assert "Moderate" in AffinityCalculator.classify_affinity(50.0)
    assert "Weak" in AffinityCalculator.classify_affinity(500.0)
    assert "Very Weak" in AffinityCalculator.classify_affinity(5000.0)


def test_ic50_to_ki():
    """Test IC50 to Ki conversion."""
    ic50 = 100.0  # nM
    substrate_conc = 10.0  # nM
    km = 50.0  # nM
    
    ki = AffinityCalculator.ic50_to_ki(ic50, substrate_conc, km)
    
    assert ki > 0
    assert ki < ic50  # Ki should be less than IC50
