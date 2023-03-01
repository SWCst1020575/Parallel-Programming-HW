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
            if (s[i] == BoxOnNormal)
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
            if (s[i] == PlayerOnly || s[i] == PlayerOnNormal || s[i] == PlayerOnOnly || s[i] == PlayerOnTarget) {
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
        delete open.top();
        open.pop();
    }
    delete currentStep;
}
bool Sokoban::isComplete(Step &nowStep) {
    for (auto i : nowStep.boxPos)
        if (map[i.second][i.first] != Target || map[i.second][i.first] != BoxOnTarget || map[i.second][i.first] != PlayerOnTarget)
            return false;
    return true;
}
bool Sokoban::isDead(Step &nowStep) {
    for (auto i : nowStep.boxPos) {
        bool up = false, left = false, down = false, right = false;  // true == obstacle
        if (map[i.second - 1][i.first] == Wall || nowStep.boxPos.find(std::make_pair(i.first, i.second - 1)) != nowStep.boxPos.end())
            up = true;
        if (map[i.second][i.first - 1] == Wall || nowStep.boxPos.find(std::make_pair(i.first - 1, i.second)) != nowStep.boxPos.end())
            left = true;
        if (map[i.second + 1][i.first] == Wall || nowStep.boxPos.find(std::make_pair(i.first, i.second + 1)) != nowStep.boxPos.end())
            down = true;
        if (map[i.second][i.first + 1] == Wall || nowStep.boxPos.find(std::make_pair(i.first + 1, i.second)) != nowStep.boxPos.end())
            right = true;
        if ((up && left) || (up && right) || (down && left) || (down && right))
            return true;
    }
    return false;
}
bool Sokoban::move(Step &nowStep, char dir) {
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
        return false;
    if (map[newPlayerPos.second][newPlayerPos.first] == Wall)
        return false;
    nowStep.playerPosX = newPlayerPos.first;
    nowStep.playerPosY = newPlayerPos.second;
    auto boxIter = nowStep.boxPos.find(newPlayerPos);
    if (boxIter != nowStep.boxPos.end()) {
        if (!moveBox(nowStep, dir, boxIter))
            return false;
    }
    nowStep.stepHistory.push_back(dir);
    return true;
}
bool Sokoban::moveBox(Step &nowStep, char dir, std::unordered_set<std::pair<int, int>>::iterator &it) {
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
    if (map[newBoxPos.second][newBoxPos.first] == Wall || nowStep.boxPos.find(newBoxPos) != nowStep.boxPos.end() || playOnlyPos.find(newBoxPos) != playOnlyPos.end())
        return false;
    nowStep.boxPos.erase(it);
    nowStep.boxPos.emplace(newBoxPos);
    DebugLog("Move box " << dir << " !!");
    return true;
}
void Sokoban::computeAstarFunction(Step &current) {
    int temp = 0;
    auto targetIter = targetPos.begin();
    for (auto i : current.boxPos) {
        if (targetIter != targetPos.end()) {
            temp += abs(i.first - targetIter->first) + abs(targetIter->first - current.playerPosX) + abs(i.second - targetIter->second) + abs(targetIter->second - current.playerPosY);
            targetIter++;
        } else
            targetIter = targetPos.begin();
    }
    current.predictCost = temp;
}
void Sokoban::findLeastCost() {
    currentStep = open.top();
    open.pop();
}
void Sokoban::printOpen() {
    auto pq = open;
    while (!pq.empty()) {
        std::cout << pq.top()->predictCost << " ";
        pq.pop();
    }
    std::cout << "\n"
              << open.size() << "\n";
}
bool Sokoban::solve() {
    if (isComplete(*currentStep))
        return true;
    if (isDead(*currentStep))
        return false;
    closed.emplace(*currentStep);
    std::list<Step *> dirSave;
    int count = 0;
    DebugLog(currentStep->playerPosX << " " << currentStep->playerPosY);
    while (!isComplete(*currentStep)) {
        auto upStep = new Step(*currentStep);
        if (move(*upStep, 'W') && !isDead(*upStep))
            dirSave.push_back(upStep);
        else
            delete upStep;
        auto leftStep = new Step(*currentStep);
        if (move(*leftStep, 'A') && !isDead(*leftStep))
            dirSave.push_back(leftStep);
        else
            delete leftStep;
        auto downStep = new Step(*currentStep);
        if (move(*downStep, 'S') && !isDead(*downStep))
            dirSave.push_back(downStep);
        else
            delete downStep;
        auto rightStep = new Step(*currentStep);
        if (move(*rightStep, 'D') && !isDead(*rightStep))
            dirSave.push_back(rightStep);
        else
            delete rightStep;

        for (auto it : dirSave)
            computeAstarFunction(*it);
        while (!dirSave.empty()) {
            open.emplace(dirSave.front());
            dirSave.pop_front();
        }
        if (open.size() > 0) {
            findLeastCost();
            while (closed.find(currentStep) != closed.end() && open.size() > 0)
                findLeastCost();
            closed.emplace(*currentStep);
        } else
            return false;  // no answer
        printOpen();
        for (auto i : currentStep->stepHistory)
            std::cout << i << " | ";
        int n;
        std::cin >> n;
    }
    for (auto i : currentStep->stepHistory)
        std::cout << i;
    std::cout << "\n";
    return true;
}