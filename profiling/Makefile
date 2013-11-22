CC = gcc
CFLAGS = -Wall
LDFLAGS = -lOpenCL
OBJDIR = ./obj
BINDIR = .
SRCDIR = ./src

all: $(BINDIR)/srs2d-physics

rebuild: clean all

clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/srs2d-physics
	@test ! -d $(OBJDIR) || rmdir --ignore-fail-on-non-empty $(OBJDIR)

$(BINDIR)/srs2d-physics: $(OBJDIR)/physics.o
	@mkdir -p $(BINDIR)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/physics.o: $(SRCDIR)/physics.c
	@mkdir -p $(OBJDIR)
	$(CC) -o $@ -c $(CFLAGS) $^