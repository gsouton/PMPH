CC = nvcc
#CFLAGS = -Werror warn
FLAGS = -O3
EXEC = wa1-task3 

$(EXEC):
	$(CC) $(FLAGS) $@.cu -o $(EXEC)

default: $(EXEC)
	./wa1-task3
	

clean:
	rm $(EXEC)
