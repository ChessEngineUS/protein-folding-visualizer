"""Plotting tools for confidence metrics and analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from pathlib import Path


class ConfidencePlotter:
    """Generate publication-quality confidence plots."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_plddt(
        self,
        plddt: np.ndarray,
        title: str = "Per-Residue Confidence (pLDDT)",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ) -> plt.Figure:
        """
        Plot pLDDT scores along the sequence.
        
        Args:
            plddt: Array of pLDDT scores
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        residues = np.arange(1, len(plddt) + 1)
        
        # Plot with color coding
        colors = []
        for score in plddt:
            if score > 90:
                colors.append('#0053D6')  # Blue
            elif score > 70:
                colors.append('#65CBF3')  # Cyan
            elif score > 50:
                colors.append('#FFDB13')  # Yellow
            else:
                colors.append('#FF7D45')  # Orange
        
        ax.bar(residues, plddt, color=colors, width=1.0, edgecolor='none')
        
        # Add horizontal lines for thresholds
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='High confidence')
        ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='Medium confidence')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Low confidence')
        
        ax.set_xlabel('Residue Position', fontsize=12)
        ax.set_ylabel('pLDDT Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pae(
        self,
        pae: np.ndarray,
        title: str = "Predicted Aligned Error (PAE)",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 9)
    ) -> plt.Figure:
        """
        Plot PAE matrix as heatmap.
        
        Args:
            pae: PAE matrix
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(
            pae,
            cmap='Greens_r',
            vmin=0,
            vmax=31.75,
            origin='lower'
        )
        
        ax.set_xlabel('Scored Residue', fontsize=12)
        ax.set_ylabel('Aligned Residue', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Expected Position Error (Å)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pae_domains(
        self,
        pae: np.ndarray,
        domain_boundaries: List[Tuple[int, int]],
        domain_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot PAE matrix with domain annotations.
        
        Args:
            pae: PAE matrix
            domain_boundaries: List of (start, end) tuples for domains
            domain_labels: Optional labels for domains
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = self.plot_pae(pae, save_path=None)
        ax = fig.axes[0]
        
        # Add domain boundaries
        for i, (start, end) in enumerate(domain_boundaries):
            color = f'C{i}'
            
            # Draw rectangles
            ax.axhline(y=start, color=color, linestyle='--', alpha=0.7)
            ax.axhline(y=end, color=color, linestyle='--', alpha=0.7)
            ax.axvline(x=start, color=color, linestyle='--', alpha=0.7)
            ax.axvline(x=end, color=color, linestyle='--', alpha=0.7)
            
            # Add label
            if domain_labels and i < len(domain_labels):
                mid = (start + end) / 2
                ax.text(
                    mid, mid, domain_labels[i],
                    ha='center', va='center',
                    color=color, fontweight='bold'
                )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confidence_summary(
        self,
        plddt: np.ndarray,
        pae: np.ndarray,
        ptm: float,
        iptm: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive confidence summary figure.
        
        Args:
            plddt: pLDDT scores
            pae: PAE matrix
            ptm: pTM score
            iptm: ipTM score (for complexes)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # pLDDT plot
        ax1 = fig.add_subplot(gs[:, 0:2])
        residues = np.arange(1, len(plddt) + 1)
        colors = ['#0053D6' if s > 90 else '#65CBF3' if s > 70 else '#FFDB13' if s > 50 else '#FF7D45' for s in plddt]
        ax1.bar(residues, plddt, color=colors, width=1.0, edgecolor='none')
        ax1.set_xlabel('Residue Position')
        ax1.set_ylabel('pLDDT Score')
        ax1.set_title('Per-Residue Confidence')
        ax1.set_ylim(0, 100)
        
        # PAE heatmap
        ax2 = fig.add_subplot(gs[:, 2])
        im = ax2.imshow(pae, cmap='Greens_r', vmin=0, vmax=31.75, origin='lower')
        ax2.set_xlabel('Scored')
        ax2.set_ylabel('Aligned')
        ax2.set_title('PAE Matrix')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        
        # Add text summary
        summary_text = f"Mean pLDDT: {np.mean(plddt):.1f}\n"
        summary_text += f"pTM: {ptm:.3f}\n"
        if iptm is not None:
            summary_text += f"ipTM: {iptm:.3f}"
        
        fig.text(0.02, 0.98, summary_text, 
                verticalalignment='top',
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
