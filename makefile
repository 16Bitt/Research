SOURCES	= learn.o 
TARGET	= learn
WAFFLES	= /home/austin/src/waffles
CFLAGS	= -L$(WAFFLES)/lib -lGClasses
CPPFLAGS	= $(CFLAGS) -std=c++11 -lpthread


all: $(SOURCES) auth.arff
	g++ $(CPPFLAGS) $(SOURCES) $(WAFFLES)/lib/libGClasses.a -o $(TARGET)

run: all
	./$(TARGET)

auth.arff:
	waffles_transform import kdd.csv > auth.arff

clean:
	-rm $(TARGET) $(SOURCES)
	rm auth.arff
