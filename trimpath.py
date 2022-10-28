import sys

file=sys.argv[1]

correctLines=[]

with open(file) as f:
    lines = f.readlines()
    skip=False
    for line in lines:
        if skip:
            skip=False
            continue
        if len(line.strip().split(' ')) == 2:
            skip=True
            continue
        #print(len(line.strip().split(' ')))
        #print(line.strip().split(' '))
        correctLines.append(line)
        
with open(file+'.fixed','w') as f:
    for line in correctLines:
        f.write(line)
