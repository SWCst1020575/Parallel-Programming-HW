#define DEBUG 1
#define MAXPIXELS 256

// define input type
#define PlayerOnNormal 'o'
#define PlayerOnTarget 'O'
#define BoxOnNormal 'x'
#define BoxOnTarget 'X'
#define Normal ' '
#define Target '.'
#define Wall '#'
#define PlayerOnly '@'
#define PlayerOnOnly '!'

#if DEBUG
#define DebugLog(x) std::cout << x << '\n'
#define PrintSokoban(x)             \
    for (int i = 0; i < h; i++) {   \
        for (int j = 0; j < w; j++) \
            std::cout << x[i][j];   \
        std::cout << '\n';          \
    }
#elif
#define DebugLog(x)
#define PrintSokoban(x)
#endif


#include <boost/functional/hash.hpp>
#include <fstream>
#include <iostream>
#include <list>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Step.h"

class Sokoban {
   public:
    Sokoban(int &, char **, std::ifstream &);
    ~Sokoban();
    bool isComplete(Step &);
    bool isDead(Step &);
    Step *move(Step &, char);
    bool moveBox(Step *, char, std::unordered_set<std::pair<int, int>>::iterator &);
    char *operator[](int index) { return map[index]; }
    void computeAstarFunction(Step &);
    void printOpen();
    void findLeastCost();
    bool solve();

   private:
    char **map;
    int totalTarget, totalBox;
    int boxOnTarget;
    Step *currentStep;
    std::unordered_set<std::pair<int, int>, PairHash> targetPos;
    std::unordered_set<std::pair<int, int>, PairHash> playOnlyPos;
    std::unordered_set<Step, stepHash> closed;
    std::unordered_set<Step, stepHash> openSave;
    std::priority_queue<Step *, std::vector<Step *>, stepCompare> open;
};
