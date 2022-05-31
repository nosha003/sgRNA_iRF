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

sgRNA file is a tab-delimited file with: chr, start, end, sgRNA, nucleotide sequence, cutting efficiency score

Final feature matrix is a tab-delimited file with rows as sgRNAs and columns as features (each sgRNA has a unique ID and cutting efficiency score

iRF input files are compiled with rows as individual sgRNAs and columns as features:
- matrix.features.txt (columns include unique ID and all features for input)
- matrix.features_overlap.txt (replicate of matrix.features.txt)
- matrix.features_overlap_noSampleIDs.txt (replicate of matrix.features.txt EXCEPT ID column is removed)
- matrix.scores.txt (columns include unique ID and cutting efficiency score)
- matrix.scores_overlap.txt (replicate of matrix.scores.txt)
- matrix.scores_overlap_noSampleIDs.txt (replicate of matrix.scores.txt EXCEPT ID column is removed)
