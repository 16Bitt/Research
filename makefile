SOURCES	= learn.o 
TARGET	= learn
WAFFLES	= ~/src/waffles
CFLAGS	= -L $(WAFFLES)/lib -lGClasses
CPPFLAGS= $(CFLAGS) -std=c++11 -lpthread -lboost_system
CC	= gcc
USER	= abbittin
SERVER	= razor.uark.edu
BROWSER	= surf

all: $(SOURCES) auth.arff
	g++ $(CPPFLAGS) $(SOURCES) $(WAFFLES)/lib/libGClassesDbg.a -o $(TARGET)

run: all
	./$(TARGET) > log.arff

auth.arff:
	waffles_transform import kdd.csv > auth.arff

clean:
	-rm $(TARGET) $(SOURCES)
	-rm auth.arff log.arff out.svg

upload:
	scp auth.arff $(USER)@$(SERVER):/scratch/$(USER)/
	scp $(TARGET) $(USER)@$(SERVER):/scratch/$(USER)/

view: all
	waffles_plot scatter log.arff -range 0 -1 100 2 red 0 2 blue 0 1 > out.svg
	$(BROWSER) out.svg
