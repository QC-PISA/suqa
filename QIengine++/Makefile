release: main.cpp
	g++ -Wall -Wextra -std=c++11 -O3 main.cpp -o main -DNDEBUG

debug: main.cpp
	g++ -Wall -Wextra -std=c++11 -g main.cpp -o main

profiling_O0: main.cpp
	g++ -Wall -Wextra -std=c++11 -O0 -g main.cpp -o main_O0 -DNDEBUG

profiling_O3: main.cpp
	g++ -Wall -Wextra -std=c++11 -O3 -g main.cpp -o main_O3 -DNDEBUG 
