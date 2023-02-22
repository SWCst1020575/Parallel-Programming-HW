#define DEBUG 1
#define MAXPIXELS 256

#if DEBUG
#define DebugLog(x) std::cout << x << "\n"
#define PrintSokoban(x)               \
    for (int i = 0; i < x.h; i++) {   \
        for (int j = 0; j < x.w; j++) \
            std::cout << x[i][j];     \
        std::cout << "\n";            \
    }
#elif
#define DebugLog(x)
#define PrintSokoban(x)
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
class Sokoban {
   public:
    Sokoban(int &, char **, std::ifstream &);
    ~Sokoban();
    char *operator[](int index) { return map[index]; }
    int h, w;

   private:
    char **map;
    int maxRow;
};

Sokoban::Sokoban(int &argc, char *argv[], std::ifstream &file) {
    std::string s;
    bool isFirst = 0;
    int row = 0;
    while (std::getline(file, s)) {
        if (!isFirst) {
            isFirst = 1;
            this->w = s.size();
            maxRow = MAXPIXELS / this->w + 1;  // max rows
            map = new char *[maxRow];
            for (int i = 0; i < maxRow; i++)
                map[i] = new char[this->w];
        }
        for (int i = 0; i < this->w; i++)
            map[row][i] = s[i];
        row++;
    }
    this->h = row;
}
Sokoban::~Sokoban() {
    for (int i = 0; i < maxRow; i++)
        delete[] map[i];
    delete[] map;
}
int main(int argc, char *argv[]) {
    if (argc != 2) {
        DebugLog("The input format should be ./hw1 {filename}.");
        return 0;
    }
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        DebugLog("The file does not exist.");
        return 0;
    }
    Sokoban sokoban(argc, argv, file);

    PrintSokoban(sokoban);
}