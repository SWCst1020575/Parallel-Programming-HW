#include "Sokoban.h"

#include <algorithm>
#include <random>
Sokoban::Sokoban(int &argc, char *argv[], std::ifstream &file) {
    std::string s;
    bool isFirst = 0;
    int row = 0;
    totalTarget = 0;
    totalBox = 0;
    boxOnTarget = 0;
    currentStep = new Step();
    isEnd = false;
    for (int i = 0; i < THREADS; i++)
        threadStep[i] = nullptr;
    while (std::getline(file, s)) {
        if (!isFirst) {
            isFirst = 1;
            w = s.size();
            maxRow = MAXPIXELS / w + 1;  // max rows
            map = new char *[maxRow];
            for (int i = 0; i < maxRow; i++)
                map[i] = new char[w];
        }
        for (int i = 0; i < w; i++) {
            map[row][i] = s[i];
            if (s[i] == BoxOnNormal || s[i] == BoxOnTarget)
                currentStep->boxPos.emplace(std::make_pair(i, row));
            if (s[i] == Target || s[i] == PlayerOnTarget || s[i] == BoxOnTarget)
                targetPos.emplace(std::make_pair(i, row));
            if (s[i] == PlayerOnly || s[i] == PlayerOnOnly)
                playOnlyPos.emplace(std::make_pair(i, row));
            if (s[i] == BoxOnNormal || s[i] == BoxOnTarget)
                totalBox++;
            if (s[i] == Target || s[i] == BoxOnTarget || s[i] == PlayerOnTarget)
                totalTarget++;
            if (s[i] == BoxOnTarget)
                boxOnTarget++;
            if (s[i] == PlayerOnNormal || s[i] == PlayerOnOnly || s[i] == PlayerOnTarget) {
                currentStep->playerPosX = i;
                currentStep->playerPosY = row;
            }
        }
        row++;
    }
    h = row;
}
Sokoban::~Sokoban() {
    for (int i = 0; i < maxRow; i++)
        delete[] map[i];
    delete[] map;
    for (int i = 0; i < THREADS; i++)
        while (!handlerList[i].empty()) {
            delete handlerList[i].top();
            handlerList[i].pop();
        }
    delete currentStep;
}
bool Sokoban::isComplete(Step *nowStep) {
    if (nowStep == nullptr)
        return false;
    for (auto i : nowStep->boxPos)
        if (map[i.second][i.first] != Target && map[i.second][i.first] != BoxOnTarget && map[i.second][i.first] != PlayerOnTarget)
            return false;
    return true;
}
Near wall;
Near box;
bool Sokoban::isDead(Step &nowStep) {
    for (auto i : nowStep.boxPos) {
        if (map[i.second][i.first] == Target || map[i.second][i.first] == BoxOnTarget || map[i.second][i.first] == PlayerOnTarget)
            continue;
        wall.setFalse();
        box.setFalse();
        if (map[i.second - 1][i.first] == Wall)
            wall.up = true;
        if (map[i.second][i.first - 1] == Wall)
            wall.left = true;
        if (map[i.second + 1][i.first] == Wall)
            wall.down = true;
        if (map[i.second][i.first + 1] == Wall)
            wall.right = true;
        if (map[i.second - 1][i.first - 1] == Wall)
            wall.upLeft = true;
        if (map[i.second - 1][i.first + 1] == Wall)
            wall.upRight = true;
        if (map[i.second + 1][i.first - 1] == Wall)
            wall.downLeft = true;
        if (map[i.second + 1][i.first + 1] == Wall)
            wall.downRight = true;
        if (nowStep.boxPos.find(std::make_pair(i.first, i.second - 1)) != nowStep.boxPos.end())
            box.up = true;
        if (nowStep.boxPos.find(std::make_pair(i.first - 1, i.second)) != nowStep.boxPos.end())
            box.left = true;
        if (nowStep.boxPos.find(std::make_pair(i.first, i.second + 1)) != nowStep.boxPos.end())
            box.down = true;
        if (nowStep.boxPos.find(std::make_pair(i.first + 1, i.second)) != nowStep.boxPos.end())
            box.right = true;
        /*if (nowStep.boxPos.find(std::make_pair(i.first - 1, i.second - 1)) != nowStep.boxPos.end())
            box.upLeft = true;
        if (nowStep.boxPos.find(std::make_pair(i.first + 1, i.second - 1)) != nowStep.boxPos.end())
            box.upRight = true;
        if (nowStep.boxPos.find(std::make_pair(i.first - 1, i.second + 1)) != nowStep.boxPos.end())
            box.downLeft = true;
        if (nowStep.boxPos.find(std::make_pair(i.first + 1, i.second + 1)) != nowStep.boxPos.end())
            box.downRight = true;*/
        if ((wall.up || wall.down) && (wall.left || wall.right))
            return true;
        if (((box.down && (wall.downLeft || wall.downRight)) || (box.up && (wall.upLeft || wall.upRight))) && (wall.left || wall.right))
            return true;
        if (((box.left && (wall.upLeft || wall.downLeft)) || (box.right && (wall.upRight || wall.downRight))) && (wall.up || wall.down))
            return true;
    }
    return false;
}
Step *Sokoban::move(Step &nowStep, char dir, bool &isMoveBox) {
    std::pair<int, int> newPlayerPos;
    isMoveBox = false;
    switch (dir) {
        case 'W':
            newPlayerPos = std::make_pair(nowStep.playerPosX, nowStep.playerPosY - 1);
            break;
        case 'A':
            newPlayerPos = std::make_pair(nowStep.playerPosX - 1, nowStep.playerPosY);
            break;
        case 'S':
            newPlayerPos = std::make_pair(nowStep.playerPosX, nowStep.playerPosY + 1);
            break;
        case 'D':
            newPlayerPos = std::make_pair(nowStep.playerPosX + 1, nowStep.playerPosY);
            break;
    }
    if (newPlayerPos.first < 0 || newPlayerPos.first > w || newPlayerPos.second < 0 || newPlayerPos.second > h)
        return nullptr;
    if (map[newPlayerPos.second][newPlayerPos.first] == Wall)
        return nullptr;
    Step *newStep;
    auto boxIter = nowStep.boxPos.find(newPlayerPos);
    if (boxIter != nowStep.boxPos.end()) {
        if (moveBox(&nowStep, dir, boxIter)) {
            newStep = new Step(nowStep);
            newStep->boxPos.erase(newStep->boxPos.find(newPlayerPos));
            std::pair<int, int> newBoxPos;
            if (dir == 'W')
                newBoxPos = std::make_pair(newPlayerPos.first, newPlayerPos.second - 1);
            else if (dir == 'A')
                newBoxPos = std::make_pair(newPlayerPos.first - 1, newPlayerPos.second);
            else if (dir == 'S')
                newBoxPos = std::make_pair(newPlayerPos.first, newPlayerPos.second + 1);
            else if (dir == 'D')
                newBoxPos = std::make_pair(newPlayerPos.first + 1, newPlayerPos.second);
            newStep->boxPos.emplace(newBoxPos);
            newStep->playerPosX = newPlayerPos.first;
            newStep->playerPosY = newPlayerPos.second;
            newStep->stepHistory.push_back(dir);
            isMoveBox = true;
            /*
                        if ((dir == 'W' || dir == 'S') && map[newBoxPos.second][newBoxPos.first - 1] == Wall && map[newBoxPos.second][newBoxPos.first + 1] == Wall) {
                            bool isMoveBoxNext = false;
                            auto nextStep = move(*newStep, dir, isMoveBoxNext);
                            if (isMoveBoxNext && nextStep != nullptr && !isDead(*nextStep)) {
                                delete newStep;
                                return nextStep;
                            }
                        } else if ((dir == 'A' || dir == 'D') && map[newBoxPos.second - 1][newBoxPos.first] == Wall && map[newBoxPos.second + 1][newBoxPos.first] == Wall) {
                            bool isMoveBoxNext = false;
                            auto nextStep = move(*newStep, dir, isMoveBoxNext);
                            if (isMoveBoxNext && nextStep != nullptr && !isDead(*nextStep)) {
                                delete newStep;
                                return nextStep;
                            }
                        }
            */
            return newStep;
        } else
            return nullptr;
    }
    newStep = new Step(nowStep);
    newStep->playerPosX = newPlayerPos.first;
    newStep->playerPosY = newPlayerPos.second;
    newStep->stepHistory.push_back(dir);
    return newStep;
}
bool Sokoban::moveBox(Step *nowStep, char dir, std::unordered_set<std::pair<int, int>>::iterator &it) {
    std::pair<int, int> newBoxPos;
    switch (dir) {
        case 'W':
            newBoxPos = std::make_pair((*it).first, (*it).second - 1);
            break;
        case 'A':
            newBoxPos = std::make_pair((*it).first - 1, (*it).second);
            break;
        case 'S':
            newBoxPos = std::make_pair((*it).first, (*it).second + 1);
            break;
        case 'D':
            newBoxPos = std::make_pair((*it).first + 1, (*it).second);
            break;
    }
    if (newBoxPos.first < 0 || newBoxPos.first > w || newBoxPos.second < 0 || newBoxPos.second > h)
        return false;
    if (map[newBoxPos.second][newBoxPos.first] == Wall || map[newBoxPos.second][newBoxPos.first] == PlayerOnly || map[newBoxPos.second][newBoxPos.first] == PlayerOnOnly || nowStep->boxPos.find(newBoxPos) != nowStep->boxPos.end())
        return false;
    return true;
}
void Sokoban::checkMoveBox(std::list<Step *> *stepList, Step *current, std::unordered_set<std::pair<int, int>, PairHash> &playerPosList, bool &isBoxMove) {
    if (isBoxMove) {
        stepList->push_back(current);
        return;
    } else if (playerPosList.find(std::make_pair(current->playerPosX, current->playerPosY)) == playerPosList.end())
        findBox(stepList, current, playerPosList);
    delete current;
}
void Sokoban::findBox(std::list<Step *> *stepList, Step *current, std::unordered_set<std::pair<int, int>, PairHash> &playerPosList) {
    playerPosList.emplace(std::make_pair(current->playerPosX, current->playerPosY));
    bool isBoxMove;
    Step *upStep = move(*current, 'W', isBoxMove);
    if (upStep != nullptr && !isDead(*upStep))
        checkMoveBox(stepList, upStep, playerPosList, isBoxMove);
    Step *leftStep = move(*current, 'A', isBoxMove);
    if (leftStep != nullptr && !isDead(*leftStep))
        checkMoveBox(stepList, leftStep, playerPosList, isBoxMove);
    Step *downStep = move(*current, 'S', isBoxMove);
    if (downStep != nullptr && !isDead(*downStep))
        checkMoveBox(stepList, downStep, playerPosList, isBoxMove);
    Step *rightStep = move(*current, 'D', isBoxMove);
    if (rightStep != nullptr && !isDead(*rightStep))
        checkMoveBox(stepList, rightStep, playerPosList, isBoxMove);
}
int Sokoban::heuristic(Step *current) {
    int totalMinDis = 0;
    for (auto i : current->boxPos) {
        int minDis = std::numeric_limits<int>::max();
        for (auto j : targetPos) {
            int dis = abs(i.first - j.first) + abs(i.second - j.second);
            if (dis < minDis)
                minDis = dis;
        }
        totalMinDis += minDis;
    }
    return totalMinDis + current->stepHistory.size() / 10;
}

void findLeastCost(Sokoban *sokoban, int &threadID) {
    if (sokoban->threadStep[threadID] != nullptr)
        delete sokoban->threadStep[threadID];
    sokoban->handlerListMutex[threadID].lock();
    if (sokoban->handlerList[threadID].empty() || sokoban->handlerList[threadID].top() == nullptr) {
        sokoban->handlerListMutex[threadID].unlock();
        for (int i = 0; i < THREADS; i++) {
            sokoban->handlerListMutex[i].lock();
            if (sokoban->handlerList[i].size() > 0 && sokoban->handlerList[i].top() != nullptr) {
                sokoban->threadStep[threadID] = sokoban->handlerList[i].top();
                sokoban->handlerList[i].pop();
                sokoban->handlerListMutex[i].unlock();
                return;
            }
            sokoban->handlerListMutex[i].unlock();
            if (i == THREADS - 1)
                i = -1;
            if (sokoban->isEnd)
                return;
        }
    }
    sokoban->threadStep[threadID] = sokoban->handlerList[threadID].top();
    sokoban->handlerList[threadID].pop();
    sokoban->handlerListMutex[threadID].unlock();
}
void threadSolve(Sokoban *sokoban, int threadID) {
    std::list<Step *> dirSave;
    std::unordered_set<std::pair<int, int>, PairHash> playerPosList;
    findLeastCost(sokoban, threadID);
    do {
        if (sokoban->isEnd)
            return;

        playerPosList.clear();
        if (sokoban->threadStep[threadID] == nullptr)
            findLeastCost(sokoban, threadID);
        sokoban->findBox(&dirSave, sokoban->threadStep[threadID], playerPosList);
        for (auto it : dirSave)
            it->predictCost = sokoban->heuristic(it);
        int dirNum = dirSave.size();

        while (!dirSave.empty()) {
            bool isFind = false;
            for (int i = 0; i < THREADS; i++) {
                sokoban->stepSaveMutex[i].lock();
                if (sokoban->stepSave[i].find(dirSave.front()) != sokoban->stepSave[i].end())
                    isFind = true;
                sokoban->stepSaveMutex[i].unlock();
                if (isFind)
                    break;
            }
            if (!isFind) {
                int nowThread = 0;
                int minListSize = std::numeric_limits<int>::max();

                for (int i = 0; i < THREADS; i++) {
                    sokoban->handlerListMutex[i].lock();
                    if (minListSize > sokoban->handlerList[i].size()) {
                        minListSize = sokoban->handlerList[i].size();
                        nowThread = i;
                    }
                    sokoban->handlerListMutex[i].unlock();
                }
                sokoban->handlerListMutex[nowThread].lock();
                sokoban->handlerList[nowThread].emplace(dirSave.front());
                sokoban->handlerListMutex[nowThread].unlock();

                sokoban->stepSaveMutex[threadID].lock();
                sokoban->stepSave[threadID].emplace(dirSave.front());
                sokoban->stepSaveMutex[threadID].unlock();
            }
            dirSave.pop_front();
            if (sokoban->isEnd)
                return;
        }
        findLeastCost(sokoban, threadID);
        if (sokoban->isEnd)
            return;

    } while (!sokoban->isComplete(sokoban->threadStep[threadID]));
    sokoban->isEnd = true;
    sokoban->currentStep = sokoban->threadStep[threadID];
}
bool Sokoban::solve() {
    if (isComplete(currentStep))
        return true;
    if (isDead(*currentStep))
        return false;
    std::list<Step *> dirSave;
    std::unordered_set<std::pair<int, int>, PairHash> playerPosList;

    playerPosList.clear();
    findBox(&dirSave, currentStep, playerPosList);
    for (auto it : dirSave)
        it->predictCost = heuristic(it);
    int nowThread = dirSave.front()->predictCost % THREADS;
    while (!dirSave.empty()) {
        bool isFind = false;
        for (int i = 0; i < THREADS; i++) {
            if (stepSave[i].find(dirSave.front()) != stepSave[i].end())
                isFind = true;
            if (isFind)
                break;
        }
        if (!isFind) {
            handlerList[nowThread].emplace(dirSave.front());
            stepSave[nowThread].emplace(dirSave.front());
        }
        dirSave.pop_front();
        nowThread = (nowThread >= THREADS - 1) ? 0 : nowThread + 1;
    }

    for (int i = 0; i < THREADS; i++)
        thread[i] = new std::thread(threadSolve, this, i);

    for (int i = 0; i < THREADS; i++)
        thread[i]->join();

    for (auto i : currentStep->stepHistory)
        std::cout << i;
    std::cout << "\n";
    return true;
}