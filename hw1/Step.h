extern int w, h, maxRow;
#include <boost/functional/hash.hpp>
#include <list>
#include <unordered_set>
#include <utility>
#include <vector>
typedef boost::hash<std::pair<int, int>> PairHash;

class Step {
   public:
    Step(){};
    Step(Step &);
    Step(Step *);
    friend bool operator>(const Step &l, const Step &r) { return l.predictCost > r.predictCost; };
    friend bool operator<(const Step &l, const Step &r) { return l.predictCost < r.predictCost; };
    bool operator()(const Step &l, const Step &r) { return l.predictCost > r.predictCost; };
    friend bool operator==(const Step &l, const Step &r) {
        if (l.playerPosX != r.playerPosX || l.playerPosY != r.playerPosY)
            return false;
        for (auto i : l.boxPos)
            if (r.boxPos.find(i) == r.boxPos.end())
                return false;
        return true;
    };
    ~Step();
    std::unordered_set<std::pair<int, int>, PairHash> boxPos;
    std::list<char> stepHistory;
    int playerPosX, playerPosY;
    int predictCost;
};
struct stepHash {
    std::size_t operator()(Step const &) const;
};
class stepCompare {
   public:
    bool operator()(const Step &a, const Step &b) {
        return a.predictCost > b.predictCost;
    }
};