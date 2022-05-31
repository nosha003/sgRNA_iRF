library(dplyr)
library(reshape2)
library(tidyr)

setwd("") # set directory to "raw" calculation feature files
structure <- read.delim("MFE.txt", header=T, sep="\t", stringsAsFactors = F) # minimum free energy from ViennaRNA
nuc <- read.delim("nuc.count.txt", header=T, sep="\t", stringsAsFactors = F) # Tm file 
score <- read.delim("score.txt", header=T, sep="\t", stringsAsFactors = F) # sgRNA cutting efficiency score file
score.df <- score[,c(1:2)] # subset file to include only unique ID and cutting efficiency score
colnames(score.df) <- c("sgRNAID", "cut.score")

structure.df <- structure[,2]
gc.df <- nuc[,7]
temp.df <- nuc[,8]

# structure, gc, temp
structure.df <- data.frame(structure[,2])
gc.df <- data.frame(nuc[,7])
temp.df <- data.frame(nuc[,8])

structure.df$scale <- "sgRNA.raw"
gc.df$scale <- "sgRNA.raw"
temp.df$scale <- "sgRNA.raw"

structure.df$sgRNAID <- structure[,1]
gc.df$sgRNAID <- nuc[,1]
temp.df$sgRNAID <- nuc[,1]

structure.temp <- left_join(structure.df, temp.df, by=c("sgRNAID", "scale"))
structure.temp.gc <- left_join(structure.temp, gc.df, by=c("sgRNAID", "scale"))
score.structure.temp.gc <- left_join(score, structure.temp.gc, by=c("sgRNAID"))
colnames(score.structure.temp.gc) <- c("sgRNAID", "cut.score", "seq", "sgRNA.structure", "scale", "sgRNA.temp", "sgRNA.gc")

## add one-hot encoding of sequence
setwd() # set directory to kmer one-hot encoding python output files
onehot.ind1 <- read.delim("ind1.txt", header=T, sep=" ")
onehot.ind2 <- read.delim("ind2.txt", header=T, sep=" ")
onehot.dep1 <- read.delim("dep1.txt", header=F, sep=" ")
onehot.dep2 <- read.delim("dep2.txt", header=F, sep=" ")
onehot.dep3 <- read.delim("dep3.txt", header=F, sep=" ")
onehot.dep4 <- read.delim("dep4.txt", header=F, sep=" ")
colnames(onehot.dep1)[1] <- "sgRNAID"
colnames(onehot.dep2)[1] <- "sgRNAID"
colnames(onehot.dep3)[1] <- "sgRNAID"
colnames(onehot.dep4)[1] <- "sgRNAID"

onehot.ind <- full_join(onehot.ind1, onehot.ind2, by="sgRNAID")
onehot.dep12 <- full_join(onehot.dep1[,1:ncol(onehot.dep1)-1], onehot.dep2[,1:ncol(onehot.dep2)-1], by="sgRNAID")
onehot.dep123 <- full_join(onehot.dep12, onehot.dep3[,1:ncol(onehot.dep3)-1], by="sgRNAID")
onehot.dep <- full_join(onehot.dep123, onehot.dep4[,1:ncol(onehot.dep4)-1], by="sgRNAID")

onehot <- full_join(onehot.ind, onehot.dep, by="sgRNAID")
onehot$scale <- "sgRNA.raw"

data.onehot <- left_join(score.structure.temp.gc, onehot, by=c("sgRNAID", "scale"))

data.onehot.melt <- melt(data.onehot[,c(1,2,4:ncol(data.onehot))], id=c("cut.score", "scale", "sgRNAID"))
data.onehot.melt.na <- na.omit(data.onehot.melt)

data.onehot.id <- data.onehot.melt.na %>% unite(feature.scale, c(variable, scale), sep = "")
data.onehot.id$value <- as.numeric(data.onehot.id$value)
data.onehot.id <- data.onehot.id[!(is.na(data.onehot.id$value) | data.onehot.id$value==""), ]
colnames(data.onehot.id) <- c("cut.score", "feature.scale", "sgRNAID", "value")

# pam (distance and nucleotide)
setwd("") # set directory to file for distance to PAM file
sgRNA.pam <- read.table("closestPAM.bed", header=F, sep="\t", stringsAsFactors = F)
sgRNA.pam.sub <- sgRNA.pam[,c(4,12,13)]
colnames(sgRNA.pam.sub) <- c("sgRNAID", "pam.code", "pam.distance")
sgRNA.pam.onehot <- sgRNA.pam.sub %>% mutate(PAM.A = ifelse(pam.code == "AGG" | pam.code == "CCT", 1, 0), PAM.C = ifelse(pam.code == "CGG" | pam.code == "CCG", 1, 0), PAM.T = ifelse(pam.code == "TGG" | pam.code == "CCA", 1, 0), PAM.G = ifelse(pam.code == "GGG" | pam.code == "CCC", 1, 0))
sgRNA.pam.df <- sgRNA.pam.onehot[,c(1,3:7)]
sgRNA.pam.df$id <- "Cas9"
sgRNA.pam.id <- unite(sgRNA.pam.df, "sgRNAID", c(sgRNAID, id), sep="_")

score.location <- left_join(score.df, sgRNA.pam.id, by="sgRNAID")
score.location$scale <- 0

score.location.melt <- melt(score.location, id=c("cut.score", "scale", "sgRNAID"))
score.location.na <- na.omit(score.location)
colnames(score.location.na) <- c("cut.score", "scale", "sgRNAID", "variable", "value")

score.location.id <- score.location.na %>% unite(feature.scale, c(variable, scale), sep = "")
df.pam.dcast <- score.location.id %>% dcast(sgRNAID + cut.score ~ feature.scale, value.var = "value", fun.aggregate=mean, na.rm=TRUE)

df.onehot.dcast <- data.onehot.id %>% dcast(sgRNAID + cut.score ~ feature.scale, value.var = "value", fun.aggregate=mean, na.rm=TRUE)
df.onehot.pam <- left_join(df.onehot.dcast, df.pam.dcast, by=c("sgRNAID"))
df.onehot.pam.na <- na.omit(df.onehot.pam)

# location relative to gene
setwd("") # set directory to file for distance to gene
sgRNA.genes <- read.table("gene.closest.bed", header=F, sep="\t", stringsAsFactors = F)
sgRNA.genes.df <- sgRNA.genes[,c(4,14)]
colnames(sgRNA.genes.df) <- c("sgRNAID", "gene.distance")
sgRNA.genes.df$id <- "Cas9"
sgRNA.genes.id <- unite(sgRNA.genes.df, "sgRNAID", c(sgRNAID, id), sep="_")

score.gene <- left_join(score.df, sgRNA.genes.id, by=c("sgRNAID"))
score.gene$scale <- 0

score.gene.melt <- melt(score.gene, id=c("cut.score", "scale", "sgRNAID"))
score.gene.melt.na <- na.omit(score.gene.melt)
colnames(score.gene.melt.na) <- c("cut.score", "scale", "sgRNAID", "variable", "value")

score.gene.id <- score.gene.melt.na %>% unite(feature.scale, c(variable, scale), sep = "")
df.gene.dcast <- score.gene.id %>% dcast(sgRNAID + cut.score ~ feature.scale, value.var = "value", fun.aggregate=mean, na.rm=TRUE)
df.gene.dcast.na <- na.omit(df.gene.dcast)

df <- inner_join(df.gene.dcast.na, df.onehot.pam.na, by=c("sgRNAID"))
write.table(df, "raw.matrix.txt", quote=F, row.names=F, sep="\t")


# add quantum chemical property features to data table
library(dplyr)
library(reshape2)

setwd("") # set to directory with sgRNA file
seq <- read.delim("sgRNA.sequence.txt", header=T, sep=" ", stringsAsFactors = F) # file with sgRNA nucleotide sequence

# Monomer QCT
setwd("") # set to directory with quantum chemical property files
tensor <- read.delim("HL.Bond.Monomer.txt", header=T, sep="\t", stringsAsFactors = F)

tensor.features <- tensor[,1]
rownames(tensor) <- tensor[,1]
tensor.df <- tensor[,2:ncol(tensor)]
tensor.t <- as.data.frame(t(tensor.df))
tensor.t$base <- names(tensor[,2:ncol(tensor)])

rownames(seq) <- seq[,1]
seq.melt <- melt(seq, id="sgRNAID")
colnames(seq.melt) <- c("sgRNAID", "position", "base")

seq.tensor <- left_join(seq.melt, tensor.t, by="base")
seq.tensor.melt <- melt(seq.tensor, id=c("sgRNAID", "position", "base"))
monomer <- dcast(seq.tensor.melt, sgRNAID ~ position + variable, value.var="value")

# Basepair QCT
tensor <- read.delim("HL.Bond.Basepair.txt", header=T, sep="\t", stringsAsFactors = F)
tensor.features <- tensor[,1]
rownames(tensor) <- tensor[,1]
tensor.df <- tensor[,2:ncol(tensor)]
tensor.t <- as.data.frame(t(tensor.df))
tensor.t$base <- names(tensor[,2:ncol(tensor)])

rownames(seq) <- seq[,1]
seq.melt <- melt(seq, id="sgRNAID")
colnames(seq.melt) <- c("sgRNAID", "position", "base")

seq.tensor <- left_join(seq.melt, tensor.t, by="base")
seq.tensor.melt <- melt(seq.tensor, id=c("sgRNAID", "position", "base"))
basepair <- dcast(seq.tensor.melt, sgRNAID ~ position + variable, value.var="value")

# Dimer QCT
tensor <- read.delim("HL.Bond.Dimer.txt", header=T, sep="\t", stringsAsFactors = F)
seq.dimer <- seq %>% unite("p1", p1:p2, remove=F, sep= "") %>% unite("p2", p2:p3, remove=F, sep= "") %>% unite("p3", p3:p4, remove=F, sep= "") %>% unite("p4", p4:p5, remove=F, sep= "") %>% unite("p5", p5:p6, remove=F, sep= "") %>% unite("p6", p6:p7, remove=F, sep= "") %>% unite("p7", p7:p8, remove=F, sep= "") %>% unite("p8", p8:p9, remove=F, sep= "") %>% unite("p9", p9:p10, remove=F, sep= "") %>% unite("p10", p10:p11, remove=F, sep= "") %>% unite("p11", p11:p12, remove=F, sep= "") %>% unite("p12", p12:p13, remove=F, sep= "") %>% unite("p13", p13:p14, remove=F, sep= "") %>% unite("p14", p14:p15, remove=F, sep= "") %>% unite("p15", p15:p16, remove=F, sep= "") %>% unite("p16", p16:p17, remove=F, sep= "") %>% unite("p17", p17:p18, remove=F, sep= "") %>% unite("p18", p18:p19, remove=F, sep= "") %>% unite("p19", p19:p20, remove=T, sep= "")

tensor.features <- tensor[,1]
rownames(tensor) <- tensor[,1]
tensor.df <- tensor[,2:ncol(tensor)]
tensor.t <- as.data.frame(t(tensor.df))
tensor.t$base <- names(tensor[,2:ncol(tensor)])

rownames(seq.dimer) <- seq.dimer[,1]
seq.df <- seq.dimer[,1:20]
seq.melt <- melt(seq.df, id="sgRNAID")
colnames(seq.melt) <- c("sgRNAID", "position", "base")

seq.tensor <- left_join(seq.melt, tensor.t, by="base")
seq.tensor.melt <- melt(seq.tensor, id=c("sgRNAID", "position", "base"))
dimer <- dcast(seq.tensor.melt, sgRNAID ~ position + variable, value.var="value")

# Trimer QCT
tensor <- read.delim("HL.Bond.Trimer.txt", header=T, sep="\t", stringsAsFactors = F)
seq.trimer <- seq %>% unite("p1", p1:p3, remove=F, sep= "") %>% unite("p2", p2:p4, remove=F, sep= "") %>% unite("p3", p3:p5, remove=F, sep= "") %>% unite("p4", p4:p6, remove=F, sep= "") %>% unite("p5", p5:p7, remove=F, sep= "") %>% unite("p6", p6:p8, remove=F, sep= "") %>% unite("p7", p7:p9, remove=F, sep= "") %>% unite("p8", p8:p10, remove=F, sep= "") %>% unite("p9", p9:p11, remove=F, sep= "") %>% unite("p10", p10:p12, remove=F, sep= "") %>% unite("p11", p11:p13, remove=F, sep= "") %>% unite("p12", p12:p14, remove=F, sep= "") %>% unite("p13", p13:p15, remove=F, sep= "") %>% unite("p14", p14:p16, remove=F, sep= "") %>% unite("p15", p15:p17, remove=F, sep= "") %>% unite("p16", p16:p18, remove=F, sep= "") %>% unite("p17", p17:p19, remove=F, sep= "") %>% unite("p18", p18:p20, remove=F, sep= "")

tensor.features <- tensor[,1]
rownames(tensor) <- tensor[,1]
tensor.df <- tensor[,2:ncol(tensor)]
tensor.t <- as.data.frame(t(tensor.df))
tensor.t$base <- names(tensor[,2:ncol(tensor)])

rownames(seq.trimer) <- seq.trimer[,1]
seq.df <- seq.trimer[,1:19]
seq.melt <- melt(seq.df, id="sgRNAID")
colnames(seq.melt) <- c("sgRNAID", "position", "base")

seq.tensor <- left_join(seq.melt, tensor.t, by="base")
seq.tensor.melt <- melt(seq.tensor, id=c("sgRNAID", "position", "base"))
trimer <- dcast(seq.tensor.melt, sgRNAID ~ position + variable, value.var="value")


# Tetramer QCT
tensor <- read.delim("HL.Bond.Tetramer.txt", header=T, sep="\t", stringsAsFactors = F)
seq.tetramer <- seq %>% unite("p1", p1:p4, remove=F, sep= "") %>% unite("p2", p2:p5, remove=F, sep= "") %>% unite("p3", p3:p6, remove=F, sep= "") %>% unite("p4", p4:p7, remove=F, sep= "") %>% unite("p5", p5:p8, remove=F, sep= "") %>% unite("p6", p6:p9, remove=F, sep= "") %>% unite("p7", p7:p10, remove=F, sep= "") %>% unite("p8", p8:p11, remove=F, sep= "") %>% unite("p9", p9:p12, remove=F, sep= "") %>% unite("p10", p10:p13, remove=F, sep= "") %>% unite("p11", p11:p14, remove=F, sep= "") %>% unite("p12", p12:p15, remove=F, sep= "") %>% unite("p13", p13:p16, remove=F, sep= "") %>% unite("p14", p14:p17, remove=F, sep= "") %>% unite("p15", p15:p18, remove=F, sep= "") %>% unite("p16", p16:p19, remove=F, sep= "") %>% unite("p17", p17:p20, remove=F, sep= "") 

tensor.features <- tensor[,1]
rownames(tensor) <- tensor[,1]
tensor.df <- tensor[,2:ncol(tensor)]
tensor.t <- as.data.frame(t(tensor.df))
tensor.t$base <- names(tensor[,2:ncol(tensor)])

rownames(seq.tetramer) <- seq.tetramer[,1]
seq.df <- seq.tetramer[,1:18]
seq.melt <- melt(seq.df, id="sgRNAID")
colnames(seq.melt) <- c("sgRNAID", "position", "base")

seq.tensor <- left_join(seq.melt, tensor.t, by="base")
seq.tensor.melt <- melt(seq.tensor, id=c("sgRNAID", "position", "base"))
tetramer <- dcast(seq.tensor.melt, sgRNAID ~ position + variable, value.var="value")

# Combine all QCT kmers
monomer.basepair <- rbind(monomer, basepair)
monomer.basepair.dimer <- rbind(monomer.basepair, dimer)
monomer.basepair.dimer.trimer <- rbind(monomer.basepair.dimer, trimer)
monomer.basepair.dimer.trimer.tetramer <- rbind(monomer.basepair.dimer.trimer, tetramer)
write.table(monomer.basepair.dimer.trimer.tetramer, "quantum.melt.txt", quote=F, row.names=F, sep="\t")


### combine raw matrix and quantum matrix to generate final feature matrix
library(dplyr)
library(reshape2)
library(tidyr)

df <- read.delim("raw.matrix.txt", header=T, sep="\t", stringsAsFactors = F)

# quantum chemical tensors
tensor <- read.delim("quantum.melt.txt", header=T, sep="\t")
tensor[is.na(tensor)] <- 0

tensor$scale <- "raw"
tensor.id <- tensor %>% unite(feature.scale, c(position, variable, scale), sep = "")
tensor.id$value <- as.numeric(tensor.id$value)
tensor.id[is.na(tensor.id)] <- 0

df.score <- unique(df[,c(1,2)])
tensor.score <- inner_join(tensor.id, df.score, by="sgRNAID")
tensor.score.order <- tensor.score[,c(5,2,1,4)]
colnames(tensor.score.order) <- c("cut.score", "feature.scale", "sgRNAID", "value")

df.dcast <- tensor.score.order %>% dcast(sgRNAID + cut.score ~ feature.scale, value.var = "value", fun.aggregate=mean, na.rm=TRUE)
df.dcast.na <- na.omit(df.dcast)
nrow(df.dcast.na)

df.location <- inner_join(df, df.dcast.na, by=c("sgRNAID"))
write.table(df.location, "finalquantum.txt", quote=F, row.names=F, sep="\t")








