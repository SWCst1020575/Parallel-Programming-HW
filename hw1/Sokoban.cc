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
bool Sokoban::isDead(Step &nowStep) {
    for (auto i : nowStep.boxPos) {
        bool up = false, left = false, down = false, right = false;  // true == obstacle
        bool upBox = false, leftBox = false, downBox = false, rightBox = false;
        if (map[i.second - 1][i.first] == Wall)
            up = true;
        if (map[i.second][i.first - 1] == Wall)
            left = true;
        if (map[i.second + 1][i.first] == Wall)
            down = true;
        if (map[i.second][i.first + 1] == Wall)
            right = true;
        if (nowStep.boxPos.find(std::make_pair(i.first, i.second - 1)) != nowStep.boxPos.end())
            upBox = true;
        if (nowStep.boxPos.find(std::make_pair(i.first - 1, i.second)) != nowStep.boxPos.end())
            leftBox = true;
        if (nowStep.boxPos.find(std::make_pair(i.first, i.second + 1)) != nowStep.boxPos.end())
            downBox = true;
        if (nowStep.boxPos.find(std::make_pair(i.first + 1, i.second)) != nowStep.boxPos.end())
            rightBox = true;
        if (map[i.second][i.first] == Target || map[i.second][i.first] == BoxOnTarget || map[i.second][i.first] == PlayerOnTarget)
            continue;
        if ((up || down) && (left || right))
            return true;
    }
    return false;
}
Step *Sokoban::move(Step &nowStep, char dir) {
    std::pair<int, int> newPlayerPos;
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
            if (dir == 'W')
                newStep->boxPos.emplace(std::make_pair(newPlayerPos.first, newPlayerPos.second - 1));
            else if (dir == 'A')
                newStep->boxPos.emplace(std::make_pair(newPlayerPos.first - 1, newPlayerPos.second));
            else if (dir == 'S')
                newStep->boxPos.emplace(std::make_pair(newPlayerPos.first, newPlayerPos.second + 1));
            else if (dir == 'D')
                newStep->boxPos.emplace(std::make_pair(newPlayerPos.first + 1, newPlayerPos.second));
            newStep->playerPosX = newPlayerPos.first;
            newStep->playerPosY = newPlayerPos.second;
            newStep->stepHistory.push_back(dir);
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
int Sokoban::heuristic(Step *current) {
    int leftBox = targetPos.size();
    int totalMinDis = 0;
    for (auto i : current->boxPos) {
        int minDis = std::numeric_limits<int>::max();
        for (auto j : targetPos) {
            int dis = abs(i.first - j.first) + abs(i.second - j.second);
            if (dis < minDis)
                minDis = dis;
            if (dis == 0)
                leftBox--;
        }
        totalMinDis += minDis;
    }
    return 2 * leftBox + totalMinDis + current->stepHistory.size() / 10;
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
    if (isComplete(*currentStep))
        return true;
    if (isDead(*currentStep))
        return false;
    closed.emplace(*currentStep);
    std::list<Step *> dirSave;
    int count = 0;

    while (!isComplete(*currentStep)) {
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
            dirSave.push_back(rightStep);

        for (auto it : dirSave)
            it->predictCost = heuristic(it);
        while (!dirSave.empty()) {
            if (closed.find(dirSave.front()) == closed.end()) 
                open.emplace(dirSave.front());
            dirSave.pop_front();
        }
        if (open.size() > 0) {
            findLeastCost();
            while (closed.find(currentStep) != closed.end() && open.size() > 0)
                findLeastCost();
            closed.emplace(*currentStep);
        } else
            return false;
        // no answer
        /*for (auto i : currentStep->stepHistory)
            std::cout << i << " | ";*/
        // DebugLog(" Now:(" << currentStep->playerPosX << "," << currentStep->playerPosY << ")");
        count++;
    }
    DebugLog("Step: " << count);
    for (auto i : currentStep->stepHistory)
        std::cout << i;
    std::cout << "\n";
    DebugLog("Open list total count: " << openSave.size());
    DebugLog("Open list count: " << open.size());
    DebugLog("Close list count: " << closed.size());
    return true;
}