smatfiles = $(wildcard *.smat)
smatgzfiles = $(wildcard *.smat.gz)

all: $(smatgzfiles:.gz=)

%.smat: %.smat.gz
	gunzip -c $< > $@

%.smat.gz: %.smat
	gzip -c $< > $@

compress: $(smatfiles:.smat=.smat.gz)
