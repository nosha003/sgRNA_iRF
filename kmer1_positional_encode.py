import os, sys
import numpy as np

onehot_dict={
  'A':'1000',
  'C':'0100',
  'T':'0010',
  'G':'0001'
}

# open input and output files
input_path = sys.argv[1]
input_file = open(input_path, 'r')
dep_file = open(input_path[:-4]+'_dependent1.txt', 'w')

# loop over nucleotide sequences
for idx, line in enumerate(input_file):

    # if first iteration, write title line
    if idx == 0:
        dep_file.writelines(line+': first-order position-dependent features'+ '\n')

    # otherwise encode sequence
    else:

        # split line by tab
        line = line.split('\t')

        # extract sequence (also remove \n)
        seq = line[-1][:-1]

        # compute position-dependent features as one-hot vectors
        pos_dep = ''.join([onehot_dict[seq[i]] for i in range(len(seq))])

        # write features to file
        dep_file.writelines(line[0] + '\t' + pos_dep + '\n')

    if idx % 10000 == 0:
        print('{0:,}'.format(idx)+' lines processed...')

print('Done!')

input_file.close()
dep_file.close()
