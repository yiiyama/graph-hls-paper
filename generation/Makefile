TARGET=generate

SOURCE=$(wildcard src/*.cc)
OBJECTS=$(patsubst src/%.cc,obj/%.o,$(SOURCE))

CFLAGS=$(shell geant4-config --cflags)
LIBS=$(shell geant4-config --libs)

main: $(TARGET)

$(TARGET): $(OBJECTS) $(TARGET).cc
	g++ $(CFLAGS) $(LIBS) -o $@ $^

visualize: $(OBJECTS) visualize.cc
	g++ $(CFLAGS) $(LIBS) -o $@ $^

obj/%.o: src/%.cc include/%.hh
	mkdir -p obj
	g++ $(CFLAGS) $(LIBS) -c -o $@ $<

.PHONY: clean

clean:
	rm -rf obj dogen
