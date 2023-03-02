#include "Step.h"
Step::Step(Step& val) {
    playerPosX = val.playerPosX;
    playerPosY = val.playerPosY;
    predictCost = val.predictCost;
    for (auto i : val.boxPos)
        boxPos.emplace(std::make_pair(i.first, i.second));
    for (auto i : val.stepHistory)
        stepHistory.push_back(i);
}
Step::Step(Step* val) {
    playerPosX = val->playerPosX;
    playerPosY = val->playerPosY;
    predictCost = val->predictCost;
    for (auto i : val->boxPos)
        boxPos.emplace(std::make_pair(i.first, i.second));
    for (auto i : val->stepHistory)
        stepHistory.push_back(i);
}
Step::~Step() {
}

std::size_t stepHash::operator()(Step const& val) const {
    std::size_t seed = 0;
    boost::hash_combine(seed, val.playerPosX);
    boost::hash_combine(seed, val.playerPosY);
    for (auto i : val.boxPos) {
        boost::hash_combine(seed, i.first);
        boost::hash_combine(seed, i.second);
    }
    return seed;
}