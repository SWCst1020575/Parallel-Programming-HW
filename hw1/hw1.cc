#include "Sokoban.h"
int w, h, maxRow;
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
    sokoban.solve();
}