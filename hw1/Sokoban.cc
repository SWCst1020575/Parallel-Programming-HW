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
    while (!open.empty()) {
        openSave.erase(*open.top());
        delete open.top();
        open.pop();
    }
    delete currentStep;
}
bool Sokoban::isComplete(Step &nowStep) {
    for (auto i : nowStep.boxPos)
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
        if (moveBox(&nowStep, dir, newPlayerPos)) {
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

            newStep->playerPosX = newPlayerPos.first;
            newStep->playerPosY = newPlayerPos.second;
            newStep->stepHistory.push_back(dir);
            isMoveBox = true;
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
bool Sokoban::moveBox(Step *nowStep, char dir, std::pair<int, int> &playerPos) {
    std::pair<int, int> newBoxPos;
    switch (dir) {
        case 'W':
            newBoxPos = std::make_pair(playerPos.first, playerPos.second - 1);
            break;
        case 'A':
            newBoxPos = std::make_pair(playerPos.first - 1, playerPos.second);
            break;
        case 'S':
            newBoxPos = std::make_pair(playerPos.first, playerPos.second + 1);
            break;
        case 'D':
            newBoxPos = std::make_pair(playerPos.first + 1, playerPos.second);
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
        if (closed.find(*current) == closed.end()) {
            stepList->push_back(current);
            return;
        } else {
            delete current;
            return;
        }
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
void Sokoban::findLeastCost() {
    delete currentStep;
    currentStep = open.top();
    open.pop();
}

bool Sokoban::solve() {
    if (isComplete(*currentStep))
        return true;
    if (isDead(*currentStep))
        return false;
    closed.emplace(currentStep);
    openSave.emplace(currentStep);
    std::list<Step *> dirSave;
    int count = 0;
    std::unordered_set<std::pair<int, int>, PairHash> playerPosList;
    do { /*
          Step *upStep = move(*currentStep, 'W');
          if (upStep != nullptr && !isDead(*upStep))
              dirSave.push_back(upStep);
          Step *leftStep = move(*currentStep, 'A');
          if (leftStep != nullptr && !isDead(*leftStep))
              dirSave.push_back(leftStep);
          Step *downStep = move(*currentStep, 'S');
          if (downStep != nullptr && !isDead(*downStep))
              dirSave.push_back(downStep);
          Step *rightStep = move(*currentStep, 'D');
          if (rightStep != nullptr && !isDead(*rightStep))
              dirSave.push_back(rightStep);*/
        playerPosList.clear();
        findBox(&dirSave, currentStep, playerPosList);
        for (auto it : dirSave)
            it->predictCost = heuristic(it);
        while (!dirSave.empty()) {
            if (openSave.find(dirSave.front()) == openSave.end()) {
                open.emplace(dirSave.front());
                openSave.emplace(dirSave.front());
            }
            dirSave.pop_front();
        }
        if (open.size() > 0) {
            findLeastCost();
            // while (closed.find(currentStep) != closed.end() && open.size() > 0)
            //   findLeastCost();
            closed.emplace(*currentStep);
        } else
            return false;
        // no answer
        /*for (auto i : currentStep->stepHistory)
            std::cout << i << " | ";*/
        // DebugLog(" Now:(" << currentStep->playerPosX << "," << currentStep->playerPosY << ")");
        count++;
    } while (!isComplete(*currentStep));
    DebugLog("Step: " << count);
    for (auto i : currentStep->stepHistory)
        std::cout << i;
    std::cout << "\n";
    DebugLog("Open list total count: " << openSave.size());
    DebugLog("Open list count: " << open.size());
    DebugLog("Close list count: " << closed.size());
    return true;
}