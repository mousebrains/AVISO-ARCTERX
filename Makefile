# Build various images
DIR = images
INPUTSALL = tpw/*.spatial.nc
INPUTSMETA = tpw/META*.spatial.nc

MONTH = 5
DOM = 5

YEARSMETA = 1993 1994 1995 1996 1997 1998 1999
YEARSMETA+= 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009
YEARSMETA+= 2010 2011 2012 2013 2014 2015 2016 2017

YEARSNRT   = 2018 2019 2020 2021 2022

ALLMETA = $(patsubst %, $(DIR)/%.all.png, $(YEARSMETA))
ALLNRT  = $(patsubst %, $(DIR)/%.all.png, $(YEARSNRT))

TwoMETA = $(patsubst %, $(DIR)/%.60.png, $(YEARSMETA))
TwoNRT  = $(patsubst %, $(DIR)/%.60.png, $(YEARSNRT))

ThreeMETA = $(patsubst %, $(DIR)/%.60.90.png, $(YEARSMETA))
ThreeNRT  = $(patsubst %, $(DIR)/%.60.90.png, $(YEARSNRT))

FourMETA = $(patsubst %, $(DIR)/%.60.90.120.png, $(YEARSMETA))
FourNRT  = $(patsubst %, $(DIR)/%.60.90.120.png, $(YEARSNRT))

all: $(ALLMETA) $(ALLNRT) $(TwoMETA) $(TwoNRT) $(ThreeMETA) $(ThreeNRT) $(FourMETA) $(FourNRT)

$(ALLMETA):
	./plot.py --month=$(MONTH) --dom=$(DOM) $(INPUTSMETA) \
		--year=$(patsubst $(DIR)/%.all.png,%, $@) \
		--png='$@'

$(ALLNRT):
	./plot.py --month=$(MONTH) --dom=$(DOM) $(INPUTSALL) \
		--year=$(patsubst $(DIR)/%.all.png,%, $@) \
		--png='$@'

$(TwoMETA):
	./plot.py --month=$(MONTH) --dom=$(DOM) $(INPUTSMETA) \
		--duration=60 \
		--year=$(patsubst $(DIR)/%.60.png,%, $@) \
		--png='$@'

$(TwoNRT):
	./plot.py --month=$(MONTH) --dom=$(DOM) $(INPUTSALL) \
		--duration=60 \
		--year=$(patsubst $(DIR)/%.60.png,%, $@) \
		--png='$@'

$(ThreeMETA):
	./plot.py --month=$(MONTH) --dom=$(DOM) $(INPUTSMETA) \
		--duration=60 --duration=90 \
		--year=$(patsubst $(DIR)/%.60.90.png,%, $@) \
		--png='$@'

$(ThreeNRT):
	./plot.py --month=$(MONTH) --dom=$(DOM) $(INPUTSALL) \
		--duration=60 --duration=90 \
		--year=$(patsubst $(DIR)/%.60.90.png,%, $@) \
		--png='$@'


$(FourMETA):
	./plot.py --month=$(MONTH) --dom=$(DOM) $(INPUTSMETA) \
		--duration=60 --duration=90 --duration=120 \
		--year=$(patsubst $(DIR)/%.60.90.120.png,%, $@) \
		--png='$@'

$(FourNRT):
	./plot.py --month=$(MONTH) --dom=$(DOM) $(INPUTSALL) \
		--duration=60 --duration=90 --duration=120 \
		--year=$(patsubst $(DIR)/%.60.90.120.png,%, $@) \
		--png='$@'

