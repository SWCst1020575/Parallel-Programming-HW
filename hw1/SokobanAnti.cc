#include <algorithm>
#include <random>

#include "Sokoban.h"
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
            if (s[i] == PlayerOnTarget || s[i] == Target || s[i] == BoxOnTarget)
                currentStep->boxPos.emplace(std::make_pair(i, row));
            if (s[i] == BoxOnNormal || s[i] == BoxOnTarget)
                targetPos.emplace(std::make_pair(i, row));
            if (s[i] == PlayerOnly || s[i] == PlayerOnOnly)
                playOnlyPos.emplace(std::make_pair(i, row));
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
        if (map[i.second][i.first] != BoxOnNormal && map[i.second][i.first] != BoxOnTarget)
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
Step *Sokoban::move(Step &nowStep, char dir) { return nullptr; }
Step *Sokoban::move(Step &nowStep, char dir, Step *boxStep) {
    std::pair<int, int> newPlayerPos;
    std::pair<int, int> boxPos;
    switch (dir) {
        case 'W':
            newPlayerPos = std::make_pair(nowStep.playerPosX, nowStep.playerPosY + 1);
            boxPos = std::make_pair(nowStep.playerPosX, nowStep.playerPosY - 1);
            break;
        case 'A':
            newPlayerPos = std::make_pair(nowStep.playerPosX + 1, nowStep.playerPosY);
            boxPos = std::make_pair(nowStep.playerPosX - 1, nowStep.playerPosY);

            break;
        case 'S':
            newPlayerPos = std::make_pair(nowStep.playerPosX, nowStep.playerPosY - 1);
            boxPos = std::make_pair(nowStep.playerPosX, nowStep.playerPosY + 1);
            break;
        case 'D':
            newPlayerPos = std::make_pair(nowStep.playerPosX - 1, nowStep.playerPosY);
            boxPos = std::make_pair(nowStep.playerPosX - 1, nowStep.playerPosY);
            break;
    }
    if (newPlayerPos.first < 0 || newPlayerPos.first > w || newPlayerPos.second < 0 || newPlayerPos.second > h)
        return nullptr;
    if (map[newPlayerPos.second][newPlayerPos.first] == Wall)
        return nullptr;
    Step *newStep;
    auto boxIter = nowStep.boxPos.find(boxPos);
    if (boxIter != nowStep.boxPos.end()) {
        if (moveBox(&nowStep, dir, boxIter)) {
            boxStep = new Step(nowStep);
            boxStep->boxPos.erase(newStep->boxPos.find(newPlayerPos));
            if (dir == 'W')
                boxStep->boxPos.emplace(std::make_pair(newPlayerPos.first, newPlayerPos.second + 1));
            else if (dir == 'A')
                boxStep->boxPos.emplace(std::make_pair(newPlayerPos.first + 1, newPlayerPos.second));
            else if (dir == 'S')
                boxStep->boxPos.emplace(std::make_pair(newPlayerPos.first, newPlayerPos.second - 1));
            else if (dir == 'D')
                boxStep->boxPos.emplace(std::make_pair(newPlayerPos.first - 1, newPlayerPos.second));
            boxStep->playerPosX = newPlayerPos.first;
            boxStep->playerPosY = newPlayerPos.second;
            boxStep->stepHistory.push_back(dir);
        }
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
            newBoxPos = std::make_pair((*it).first, (*it).second + 1);
            break;
        case 'A':
            newBoxPos = std::make_pair((*it).first + 1, (*it).second);
            break;
        case 'S':
            newBoxPos = std::make_pair((*it).first, (*it).second - 1);
            break;
        case 'D':
            newBoxPos = std::make_pair((*it).first - 1, (*it).second);
            break;
    }
    if (newBoxPos.first < 0 || newBoxPos.first > w || newBoxPos.second < 0 || newBoxPos.second > h)
        return false;
    if (map[newBoxPos.second][newBoxPos.first] == PlayerOnly || map[newBoxPos.second][newBoxPos.first] == PlayerOnOnly)
        return false;
    return true;
}
int Sokoban::heuristic(Step *current) {
    int totalMinDis = 0;
    int minPlayerDis = std::numeric_limits<int>::max();
    for (auto i : current->boxPos) {
        int minDis = std::numeric_limits<int>::max();
        for (auto j : targetPos) {
            int dis = abs(i.first - j.first) + abs(i.second - j.second);
            if (dis < minDis)
                minDis = dis;
        }
        int playerDis = abs(i.first - current->playerPosX) + abs(i.second - current->playerPosY);
        if (playerDis < minPlayerDis)
            minPlayerDis = playerDis;
        totalMinDis += minDis;
    }
    return minPlayerDis * current->boxPos.size() + totalMinDis + current->stepHistory.size() / 10;
}
void Sokoban::findLeastCost() {
    delete currentStep;
    currentStep = open.top();
    open.pop();
}
void Sokoban::printOpen() {
    DebugLog("PrintOpen Function");
    auto pq = open;
    while (!pq.empty()) {
        std::cout << pq.top()->predictCost << " ";
        pq.pop();
    }
    DebugLog("");
    DebugLog(open.size());
    DebugLog("---------Box---------");
    for (auto i : currentStep->boxPos)
        DebugLog(i.first << " " << i.second);
    DebugLog("---------Box---------");
}
bool Sokoban::solve() {
    std::list<Step *> dirSave;
    int count = 0;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (map[i][j] == Normal || map[i][j] == PlayerOnNormal || map[i][j] == PlayerOnOnly || map[i][j] == PlayerOnly || map[i][j] == BoxOnNormal) {
                auto newStep = new Step(currentStep);
                newStep->playerPosX = j;
                newStep->playerPosY = i;
                heuristic(newStep);
                open.emplace(newStep);
                openSave.emplace(currentStep);
            }
        }
    }
    findLeastCost();
    while (!isComplete(*currentStep)) {
        Step *boxStep = nullptr;
        Step *upStep = move(*currentStep, 'W', boxStep);
        if (upStep != nullptr)
            dirSave.push_back(upStep);
        if (boxStep != nullptr)
            dirSave.push_back(boxStep);
        boxStep = nullptr;
        Step *leftStep = move(*currentStep, 'A');
        if (leftStep != nullptr)
            dirSave.push_back(leftStep);
        if (boxStep != nullptr)
            dirSave.push_back(boxStep);
        boxStep = nullptr;
        Step *downStep = move(*currentStep, 'S');
        if (downStep != nullptr)
            dirSave.push_back(downStep);
        if (boxStep != nullptr)
            dirSave.push_back(boxStep);
        boxStep = nullptr;
        Step *rightStep = move(*currentStep, 'D');
        if (rightStep != nullptr)
            dirSave.push_back(rightStep);
        if (boxStep != nullptr)
            dirSave.push_back(boxStep);
        boxStep = nullptr;

        for (auto it : dirSave)
            it->predictCost = heuristic(it);
        while (!dirSave.empty()) {
            if (openSave.find(dirSave.front()) == openSave.end()) {
                openSave.emplace(dirSave.front());
                open.emplace(dirSave.front());
            }
            dirSave.pop_front();
        }
        if (open.size() > 0) {
            findLeastCost();
            while (closed.find(currentStep) != closed.end() && open.size() > 0)
                findLeastCost();
            closed.emplace(*currentStep);
        }
        // no answer
        /*for (auto i : currentStep->stepHistory)
            std::cout << i << " | ";*/
        // DebugLog(" Now:(" << currentStep->playerPosX << "," << currentStep->playerPosY << ")");
        count++;
    }
    DebugLog("Step: " << count);
    auto it = currentStep->stepHistory.end();
    while (true) {
        it--;
        std::cout << *it;
        if (it == currentStep->stepHistory.begin())
            break;
    }
    std::cout << "\n";
    DebugLog("Open list total count: " << openSave.size());
    DebugLog("Open list count: " << open.size());
    DebugLog("Close list count: " << closed.size());
    return true;
}