"""Utility functions for AlphaFold 3 predictions."""

from typing import Dict, List, Tuple
from pathlib import Path
import re


def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """
    Parse FASTA file into dictionary.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate protein sequence.
    
    Args:
        sequence: Protein sequence string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Remove whitespace
    sequence = sequence.upper().replace(' ', '')
    
    # Check for invalid characters
    invalid_chars = set(sequence) - valid_amino_acids
    if invalid_chars:
        return False, f"Invalid amino acids: {invalid_chars}"
    
    # Check minimum length
    if len(sequence) < 10:
        return False, "Sequence too short (minimum 10 residues)"
    
    # Check maximum length
    if len(sequence) > 10000:
        return False, "Sequence too long (maximum 10000 residues)"
    
    return True, "Valid"


def write_fasta(sequences: Dict[str, str], output_path: str):
    """
    Write sequences to FASTA file.
    
    Args:
        sequences: Dictionary mapping IDs to sequences
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for seq_id, sequence in sequences.items():
            f.write(f">{seq_id}\n")
            # Write 80 characters per line
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + "\n")
