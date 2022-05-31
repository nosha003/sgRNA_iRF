import os, sys
import numpy as np

onehot_dict = {
'AAA':'1000000000000000000000000000000000000000000000000000000000000000',
'AAC':'0100000000000000000000000000000000000000000000000000000000000000',
'AAT':'0010000000000000000000000000000000000000000000000000000000000000',
'AAG':'0001000000000000000000000000000000000000000000000000000000000000',
'ACA':'0000100000000000000000000000000000000000000000000000000000000000',
'ACC':'0000010000000000000000000000000000000000000000000000000000000000',
'ACT':'0000001000000000000000000000000000000000000000000000000000000000',
'ACG':'0000000100000000000000000000000000000000000000000000000000000000',
'ATA':'0000000010000000000000000000000000000000000000000000000000000000',
'ATC':'0000000001000000000000000000000000000000000000000000000000000000',
'ATT':'0000000000100000000000000000000000000000000000000000000000000000',
'ATG':'0000000000010000000000000000000000000000000000000000000000000000',
'AGA':'0000000000001000000000000000000000000000000000000000000000000000',
'AGC':'0000000000000100000000000000000000000000000000000000000000000000',
'AGT':'0000000000000010000000000000000000000000000000000000000000000000',
'AGG':'0000000000000001000000000000000000000000000000000000000000000000',
'CAA':'0000000000000000100000000000000000000000000000000000000000000000',
'CAC':'0000000000000000010000000000000000000000000000000000000000000000',
'CAT':'0000000000000000001000000000000000000000000000000000000000000000',
'CAG':'0000000000000000000100000000000000000000000000000000000000000000',
'CCA':'0000000000000000000010000000000000000000000000000000000000000000',
'CCC':'0000000000000000000001000000000000000000000000000000000000000000',
'CCT':'0000000000000000000000100000000000000000000000000000000000000000',
'CCG':'0000000000000000000000010000000000000000000000000000000000000000',
'CTA':'0000000000000000000000001000000000000000000000000000000000000000',
'CTC':'0000000000000000000000000100000000000000000000000000000000000000',
'CTT':'0000000000000000000000000010000000000000000000000000000000000000',
'CTG':'0000000000000000000000000001000000000000000000000000000000000000',
'CGA':'0000000000000000000000000000100000000000000000000000000000000000',
'CGC':'0000000000000000000000000000010000000000000000000000000000000000',
'CGT':'0000000000000000000000000000001000000000000000000000000000000000',
'CGG':'0000000000000000000000000000000100000000000000000000000000000000',
'TAA':'0000000000000000000000000000000010000000000000000000000000000000',
'TAC':'0000000000000000000000000000000001000000000000000000000000000000',
'TAT':'0000000000000000000000000000000000100000000000000000000000000000',
'TAG':'0000000000000000000000000000000000010000000000000000000000000000',
'TCA':'0000000000000000000000000000000000001000000000000000000000000000',
'TCC':'0000000000000000000000000000000000000100000000000000000000000000',
'TCT':'0000000000000000000000000000000000000010000000000000000000000000',
'TCG':'0000000000000000000000000000000000000001000000000000000000000000',
'TTA':'0000000000000000000000000000000000000000100000000000000000000000',
'TTC':'0000000000000000000000000000000000000000010000000000000000000000',
'TTT':'0000000000000000000000000000000000000000001000000000000000000000',
'TTG':'0000000000000000000000000000000000000000000100000000000000000000',
'TGA':'0000000000000000000000000000000000000000000010000000000000000000',
'TGC':'0000000000000000000000000000000000000000000001000000000000000000',
'TGT':'0000000000000000000000000000000000000000000000100000000000000000',
'TGG':'0000000000000000000000000000000000000000000000010000000000000000',
'GAA':'0000000000000000000000000000000000000000000000001000000000000000',
'GAC':'0000000000000000000000000000000000000000000000000100000000000000',
'GAT':'0000000000000000000000000000000000000000000000000010000000000000',
'GAG':'0000000000000000000000000000000000000000000000000001000000000000',
'GCA':'0000000000000000000000000000000000000000000000000000100000000000',
'GCC':'0000000000000000000000000000000000000000000000000000010000000000',
'GCT':'0000000000000000000000000000000000000000000000000000001000000000',
'GCG':'0000000000000000000000000000000000000000000000000000000100000000',
'GTA':'0000000000000000000000000000000000000000000000000000000010000000',
'GTC':'0000000000000000000000000000000000000000000000000000000001000000',
'GTT':'0000000000000000000000000000000000000000000000000000000000100000',
'GTG':'0000000000000000000000000000000000000000000000000000000000010000',
'GGA':'0000000000000000000000000000000000000000000000000000000000001000',
'GGC':'0000000000000000000000000000000000000000000000000000000000000100',
'GGT':'0000000000000000000000000000000000000000000000000000000000000010',
'GGG':'0000000000000000000000000000000000000000000000000000000000000001'
}

# open input and output files
input_path = sys.argv[1]
input_file = open(input_path, 'r')
dep_file = open(input_path[:-4]+'_dependent3.txt', 'w')

# loop over nucleotide sequences
for idx, line in enumerate(input_file):

    # if first iteration, write title line
    if idx == 0:
        dep_file.writelines(line+': third-order position-dependent features'+ '\n')

    # otherwise encode sequence
    else:

        # split line by tab
        line = line.split('\t')

        # extract sequence (also remove \n)
        seq = line[-1][:-1]

        # compute position-dependent features as one-hot vectors
        pos_dep = ''.join([onehot_dict[seq[i:i+3]] for i in range(len(seq)-2)])

        # write features to file
        dep_file.writelines(line[0] + '\t' + pos_dep + '\n')

    if idx % 10000 == 0:
        print('{0:,}'.format(idx)+' lines processed...')

print('Done!')

input_file.close()
dep_file.close()