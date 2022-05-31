## Temperature of melting (Tm)

input_file = open('ecoli.gRNA.fasta', 'r')
output_file = open('nucleotide_counts_sgRNA.tsv','w')
output_file.write('Window\tA\tC\tG\tT\tLength\tCG%\n')
from Bio import SeqIO
for cur_record in SeqIO.parse(input_file, "fasta") :
    gene_name = cur_record.name
    A_count = cur_record.seq.count('A')
    C_count = cur_record.seq.count('C')
    G_count = cur_record.seq.count('G')
    T_count = cur_record.seq.count('T')
    length = len(cur_record.seq)
    cg_percentage = float(C_count + G_count) / length
    output_line = '%s\t%i\t%i\t%i\t%i\t%i\t%f\n' % \
    (gene_name, A_count, C_count, G_count, T_count, length, cg_percentage)
    output_file.write(output_line)
    
output_file.close()
input_file.close()
exit()

