"""Tests for AlphaFold 3 predictor."""

import pytest
import numpy as np
from pathlib import Path
from src.alphafold3 import AlphaFold3Predictor, parse_fasta, validate_sequence


def test_parse_fasta(tmp_path):
    """Test FASTA file parsing."""
    fasta_content = ">protein1\nMKFLKFSLLT\n>protein2\nAVLLSVVFAF\n"
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(fasta_content)
    
    sequences = parse_fasta(str(fasta_file))
    
    assert len(sequences) == 2
    assert sequences["protein1"] == "MKFLKFSLLT"
    assert sequences["protein2"] == "AVLLSVVFAF"


def test_validate_sequence():
    """Test sequence validation."""
    # Valid sequence
    is_valid, msg = validate_sequence("MKFLKFSLLTAVLLSVVFAFSSCGDDDD")
    assert is_valid
    assert msg == "Valid"
    
    # Too short
    is_valid, msg = validate_sequence("MKFL")
    assert not is_valid
    assert "too short" in msg.lower()
    
    # Invalid characters
    is_valid, msg = validate_sequence("MKFLKFSLLTXYZ")
    assert not is_valid
    assert "invalid" in msg.lower()


def test_af3_result_dataclass():
    """Test AF3Result dataclass."""
    from src.alphafold3 import AF3Result
    
    result = AF3Result(
        pdb_path="test.pdb",
        plddt=np.array([90.0, 85.0, 95.0]),
        pae=np.array([[0, 5], [5, 0]]),
        ptm=0.85,
        iptm=None,
        ranking_confidence=0.9,
        metadata={}
    )
    
    assert result.pdb_path == "test.pdb"
    assert result.ptm == 0.85
    assert len(result.plddt) == 3
