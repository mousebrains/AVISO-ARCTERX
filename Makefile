# Parallized building of joined files and dataframe for classification
#
# Jan-2023, Pat Welch, pat@mousebrains.com

POLYGON = subset.polygon.yaml

MONTHDOM = 201
PREDAYS = 10

SRCDIR = data
JOINDIR = tpw.joined.$(MONTHDOM)
DFDIR = tpw.dataframe

SRCFILES = $(wildcard $(SRCDIR)/Eddy*.nc) $(wildcard $(SRCDIR)/META*long*.nc)
JOINFILES = $(addprefix $(JOINDIR)/, $(notdir $(SRCFILES)))

DFFILE = $(DFDIR)/dataframe.$(MONTHDOM).nc

.PHONY: all

all: $(DFFILE)

$(DFFILE):: make.dataframe.py
$(DFFILE):: $(JOINFILES)
	./make.dataframe.py --output=$@ --monthDOM=$(MONTHDOM) --preDays=$(PREDAYS) $^

$(JOINDIR)/%.nc: $(SRCDIR)/%.nc $(POLYGON) make.joined.py
	./make.joined.py --output=$(dir $@) --polygon=$(POLYGON) --monthDOM=$(MONTHDOM) $<
