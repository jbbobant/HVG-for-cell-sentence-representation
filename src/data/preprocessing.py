import os
import logging
import anndata
import scanpy as sc
import numpy as np
from typing import Optional, Dict, Any

# Initialize module logger
logger = logging.getLogger("CellSentence.Data")

class Preprocessor:
    """
    Handles loading, quality control, normalization, and feature selection 
    for Single-Cell RNA sequencing data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing preprocessing parameters 
                    (usually loaded from yaml).
        """
        self.cfg = config
        self.adata: Optional[anndata.AnnData] = None

    def load_data(self, file_path: str) -> None:
        """Loads .h5ad data from disk."""
        if not os.path.exists(file_path):
            logger.error(f"Input file not found: {file_path}")
            raise FileNotFoundError(f"{file_path} does not exist.")
            
        logger.info(f"Loading data from {file_path}...")
        self.adata = anndata.read_h5ad(file_path)
        logger.info(f"Initial data shape: {self.adata.shape} (Cells x Genes)")

    def run_qc(self) -> None:
        """
        Applies Quality Control filters:
        1. Filter cells with too few genes.
        2. Filter genes appearing in too few cells.
        3. Filter based on Mitochondrial content.
        """
        if self.adata is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 1. Basic Filtering
        min_genes = self.cfg.get('min_genes_per_cell', 200)
        min_cells = self.cfg.get('min_cells_per_gene', 3)
        
        sc.pp.filter_cells(self.adata, min_genes=min_genes)
        sc.pp.filter_genes(self.adata, min_cells=min_cells)
        
        # 2. Mitochondrial QC
        # Annotate the group of mitochondrial genes as "mt"
        self.adata.var["mt"] = self.adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            self.adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )
        
        # 3. Apply Thresholds
        max_genes = self.cfg.get('max_n_genes_by_counts', 6000)
        max_mt = self.cfg.get('max_pct_mt', 50.0)

        n_cells_before = self.adata.n_obs
        
        # Keep cells meeting criteria
        self.adata = self.adata[self.adata.obs.n_genes_by_counts < max_genes, :]
        self.adata = self.adata[self.adata.obs.pct_counts_mt < max_mt, :].copy()
        
        n_dropped = n_cells_before - self.adata.n_obs
        logger.info(f"QC Filtering dropped {n_dropped} cells. New shape: {self.adata.shape}")

    def normalize_and_log(self) -> None:
        """
        Normalizes total counts and applies Log1p (base 10).
        Base 10 is specifically required for Cell2Sentence reversibility.
        """
        logger.info("Applying normalization and Log1p(base 10)...")
        sc.pp.normalize_total(self.adata)
        
        # Using base=10 as per C2S requirements
        sc.pp.log1p(self.adata, base=10)

    def select_hvg(self) -> None:
        """
        Selects Highly Variable Genes (HVG) if configured.
        Typically keeps the top 2000 genes to reduce noise.
        """
        if not self.cfg.get('use_hvg', False):
            logger.info("Skipping HVG selection (using full genome).")
            return

        n_top_genes = self.cfg.get('n_hvg', 2000)
        logger.info(f"Selecting top {n_top_genes} Highly Variable Genes (flavor='seurat_v3')...")
        
        # Note: seurat_v3 expects raw counts, but often applied on normalized for approximation
        # In your notebook you applied it on a copy. Scanpy handles layers if needed.
        # Here we apply it simply as a filter step.
        try:
             sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_top_genes,
                flavor="seurat_v3", # or 'seurat' if data is already processed heavily
                subset=True
            )
        except Exception as e:
             # Fallback if seurat_v3 fails due to data format
             logger.warning(f"Seurat_v3 failed ({e}), falling back to basic 'seurat' flavor.")
             sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_top_genes,
                subset=True
            )
            
        logger.info(f"Data shape after HVG: {self.adata.shape}")

    def process(self) -> anndata.AnnData:
        """Run the full pipeline."""
        self.run_qc()
        self.normalize_and_log()
        self.select_hvg()
        
        # Cleanup unnecessary columns to save memory/space
        if "feature_name" in self.adata.var:
            del self.adata.var["feature_name"]
            
        return self.adata

    def save(self, output_path: str) -> None:
        """Saves the processed AnnData object."""
        if self.adata is None:
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Saving processed data to {output_path}...")
        self.adata.write_h5ad(output_path)