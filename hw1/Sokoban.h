#define DEBUG 1
#define MAXPIXELS 256
#define THREADS 6

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

#include <algorithm>
#include <atomic>
#include <boost/functional/hash.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <mutex>
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
    bool isComplete(Step *);
    bool isDead(Step &);
    Step *move(Step &, char, bool &);
    bool moveBox(Step *, char, std::unordered_set<std::pair<int, int>>::iterator &);
    char *operator[](int index) { return map[index]; }
    int heuristic(Step *);
    void checkMoveBox(std::list<Step *> *, Step *, std::unordered_set<std::pair<int, int>, PairHash> &, bool &);
    void findBox(std::list<Step *> *, Step *, std::unordered_set<std::pair<int, int>, PairHash> &);
    bool solve();

    char **map;
    int totalTarget, totalBox;
    int boxOnTarget;
    bool isEnd;
    Step *currentStep;

    Step *threadStep[THREADS];
    std::thread *thread[THREADS];
    std::mutex handlerListMutex[THREADS];
    std::mutex stepSaveMutex[THREADS];
    std::priority_queue<Step *, std::vector<Step *>, stepCompare> handlerList[THREADS];
    std::unordered_set<Step, stepHash> stepSave[THREADS];

    std::unordered_set<std::pair<int, int>, PairHash> targetPos;
    std::unordered_set<std::pair<int, int>, PairHash> playOnlyPos;
};
class Near {
   public:
    Near(){};
    ~Near(){};
    void setFalse() {
        up = false, left = false, down = false, right = false;
        upLeft = false, upRight = false, downLeft = false, downRight = false;
    }
    bool up, left, down, right;
    bool upLeft, upRight, downLeft, downRight;
};