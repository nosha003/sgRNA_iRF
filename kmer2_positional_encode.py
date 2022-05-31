import os, sys
import numpy as np

onehot_dict = {
  'AA':'1000000000000000',
  'AC':'0100000000000000',
  'AT':'0010000000000000',
  'AG':'0001000000000000',
  'CA':'0000100000000000',
  'CC':'0000010000000000',
  'CT':'0000001000000000',
  'CG':'0000000100000000',
  'TA':'0000000010000000',
  'TC':'0000000001000000',
  'TT':'0000000000100000',
  'TG':'0000000000010000',
  'GA':'0000000000001000',
  'GC':'0000000000000100',
  'GT':'0000000000000010',
  'GG':'0000000000000001'
}

# open input and output files
input_path = sys.argv[1]
input_file = open(input_path, 'r')
dep_file = open(input_path[:-4]+'_dependent2.txt', 'w')

# loop over nucleotide sequences
for idx, line in enumerate(input_file):

    # if first iteration, write title line
    if idx == 0:
        dep_file.writelines(line+': second-order position-dependent features'+ '\n')

    # otherwise encode sequence
    else:

        # split line by tab
        line = line.split('\t')

        # extract sequence (also remove \n)
        seq = line[-1][:-1]

        # compute position-dependent features as one-hot vectors
        pos_dep = ''.join([onehot_dict[seq[i:i+2]] for i in range(len(seq)-1)])

        # write features to file
        dep_file.writelines(line[0] + '\t' + pos_dep + '\n')

    if idx % 10000 == 0:
        print('{0:,}'.format(idx)+' lines processed...')

print('Done!')

input_file.close()
dep_file.close()

#/gpfs/alpine/syb105/proj-shared/Personal/noshayjm/projects/seed/kmer2_positional_encode.py
#python file.py data.txt
