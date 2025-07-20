#pragma once
#include <vector>

namespace Map {
    using mapType = std::vector<std::vector<bool>>;  // 地图类型，二维布尔数组表示障碍物
}

namespace Path {

    enum SubDivLevel {
        minSize = 0,
        midSize = 1,
        maxSize = 2
    };

    struct Sector {
        double start;  // 扇区起始角度（度）
        double end;    // 扇区结束角度（度）
    };

    struct Point {
        int x;
        int y;
    };

    struct Movement {
        double angle;
        double distance;  // 移动距离
    };

    // 使用射线投影法寻找清晰扇区
    std::vector<Sector> findSectors(const Map::mapType& _map, Point _endPoint, SubDivLevel searchLevel = minSize);

    // 主要的路径搜索函数
    std::vector<Movement> SearchPathExport(const Map::mapType& gameMap, Point start, Point end,
                                   int carRadius, int fireRadius, SubDivLevel searchLevel = minSize);

};