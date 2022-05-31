# sgRNA_iRF
A panel of sgRNA sequences are encoded into a feature matrix. This feature set is then used to train an XAI iRF model.  Output metrics including R2, Pearson correlation, Feature Importance, Feature Effect, and Sample Influence are calculated.

Feature set includes:
- Temperature of melting (Tm)
- GC conent
- Minimum Free Energy (MFE; calculated with ViennaRNA)
- Distance to PAM
- Distance to closest gene
- PAM nucleotide encoding
- Positional encoding (position-independent monomer & dimer, position-dependent monomer, dimer, trimer, tetramer)
- Quantum chemical tensors (position-dependent monomer, basepair, dimer, trimer, tetramer)


