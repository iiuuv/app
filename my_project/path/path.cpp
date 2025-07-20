#include "path.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Path {

    // 根据搜索级别计算步进大小
    double getStepSizeForLevel(SubDivLevel level) {
        switch (level) {
            case maxSize: return 2.0;   // 粗搜索
            case midSize: return 1.0;   // 中等精度
            case minSize: return 0.5;   // 精细搜索
            default: return 1.0;
        }
    }

    // 根据搜索级别调整射线投射精度
    double getRayStepSizeForLevel(SubDivLevel level) {
        switch (level) {
            case maxSize: return 1.0;   // 粗射线
            case midSize: return 0.5;   // 中等射线
            case minSize: return 0.3;   // 精细射线
            default: return 0.5;
        }
    }

    struct ObstacleRay {
        int angle;          // 角度（度）
        double distance;    // 到障碍物的距离
        bool hitObstacle;   // 是否撞到障碍物
    };

    std::vector<Sector> findSectors(const Map::mapType& _map, const Point _endPoint, SubDivLevel searchLevel,double castDistance) {
        if (_map.empty() || _map[0].empty()) {
            return {};
        }

        const int height = static_cast<int>(_map.size());
        const int width = static_cast<int>(_map[0].size());
        const double stepSize = getRayStepSizeForLevel(searchLevel);  // 根据搜索级别调整射线步进精度
        const double maxRayDistance = castDistance * 2;  // 最大射线距离

        std::vector<ObstacleRay> obstacleRays;

        // 向360度各个方向投射射线（每1度一条）
        for (int angleDeg = 0; angleDeg < 360; ++angleDeg) {
            const double angleRad = angleDeg * M_PI / 180.0;
            const double dx = std::cos(angleRad);
            const double dy = std::sin(angleRad);

            bool hitObstacle = false;
            double hitDistance = maxRayDistance;

            // 沿射线方向逐步检测
            for (int step = 0; step < static_cast<int>(maxRayDistance / stepSize); ++step) {
                const double rayDistance = step * stepSize;
                const double rayX = _endPoint.x + dx * rayDistance;
                const double rayY = _endPoint.y + dy * rayDistance;

                // 检查地图边界
                if (rayX < 0 || rayX >= width || rayY < 0 || rayY >= height) {
                    hitDistance = rayDistance;
                    hitObstacle = true;
                    break;
                }

                // 检查是否撞到障碍物
                const int gridX = static_cast<int>(rayX);
                const int gridY = static_cast<int>(rayY);

                if (gridX >= 0 && gridX < width && gridY >= 0 && gridY < height) {
                    if (_map[gridY][gridX]) {  // 撞到障碍物
                        hitDistance = rayDistance;
                        hitObstacle = true;
                        break;
                    }
                }
            }

            obstacleRays.push_back({angleDeg, hitDistance, hitObstacle});
        }

        // 分析障碍物射线，找出清晰扇区
        std::vector<Sector> clearSectors;
        const double minGapDegrees = 45.0;  // 最小间隙角度
        // 找出在关心范围内的障碍物角度
        std::vector<int> obstacleAngles;
        for (const auto& ray : obstacleRays) {
            if (ray.hitObstacle && ray.distance <= castDistance) {
                obstacleAngles.push_back(ray.angle);
            }
        }

        if (obstacleAngles.empty()) {
            // 没有近距离障碍物，整个圆周都是清晰的
            clearSectors.push_back({0.0, 360.0});
            return clearSectors;
        }

        // 排序障碍物角度
        std::sort(obstacleAngles.begin(), obstacleAngles.end());

        // 寻找相邻障碍物间的间隙
        for (size_t i = 0; i < obstacleAngles.size(); ++i) {
            const int currentAngle = obstacleAngles[i];
            const int nextAngle = obstacleAngles[(i + 1) % obstacleAngles.size()];

            double gap;
            double sectorStart, sectorEnd;

            if (nextAngle > currentAngle) {
                // 正常情况
                gap = nextAngle - currentAngle;
                sectorStart = currentAngle;
                sectorEnd = nextAngle;
            } else {
                // 处理跨越0度的情况（如从350度到10度）
                gap = (360 - currentAngle) + nextAngle;
                sectorStart = currentAngle;
                sectorEnd = nextAngle;
            }

            // 如果间隙足够大，这就是一个清晰扇区
            if (gap >= minGapDegrees) {
                if (sectorEnd > 360) {
                    // 处理跨越边界的情况
                    clearSectors.push_back({sectorStart, 360.0});
                    clearSectors.push_back({0.0, sectorEnd - 360.0});
                } else {
                    clearSectors.push_back({sectorStart, sectorEnd});
                }
            }
        }

        return clearSectors;
    }



    // 检查位置是否有效（考虑车辆半径和禁入区域）
    bool isValidPosition(const Map::mapType& _map,const Point _start,const Point _end,const int _carRadius,const int _fireRadius) {
        const int height = static_cast<int>(_map.size());
        const int width = static_cast<int>(_map[0].size());

        // 检查边界约束
        if (_start.x - _carRadius < 0 || _start.y - _carRadius < 0 ||
            _start.x + _carRadius >= width || _start.y + _carRadius >= height) {
            return false;
        }

        // 检查禁入区域：车辆不能进入距离目标点小于fireRadius的区域
        const double distToEnd = std::sqrt((_start.x - _end.x) * (_start.x - _end.x) +
                                          (_start.y - _end.y) * (_start.y - _end.y));
        if (distToEnd < _fireRadius) {
            return false;
        }

        // 检查车辆半径范围内的圆形区域是否有障碍物
        for (int dy = -_carRadius; dy <= _carRadius; ++dy) {
            for (int dx = -_carRadius; dx <= _carRadius; ++dx) {
                if (dx * dx + dy * dy <= _carRadius * _carRadius) {  // 圆形碰撞检测
                    const int checkX = _start.x + dx;
                    if (const int checkY = _start.y + dy; checkX >= 0 && checkX < width && checkY >= 0 && checkY < height) {
                        if (_map[checkY][checkX]) {  // 撞到障碍物
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    // 检查位置是否在清晰扇区内
    bool isPositionInClearSector(Point pos, Point end, const std::vector<Sector>& clearSectors) {
        if (clearSectors.empty()) {
            return false;
        }

        const double dx = pos.x - end.x;
        const double dy = pos.y - end.y;
        double angle = std::atan2(dy, dx) * 180.0 / M_PI;
        if (angle < 0) angle += 360.0;  // 归一化到[0, 360)

        // 检查每个扇区
        for (const auto& sector : clearSectors) {
            if (sector.end <= 360.0) {
                // 正常扇区
                if (angle >= sector.start && angle <= sector.end) {
                    return true;
                }
            } else {
                // 跨越0度的扇区
                if (angle >= sector.start || angle <= (sector.end - 360.0)) {
                    return true;
                }
            }
        }
        return false;
    }

    // 获取有效的目标位置
    std::vector<Point> getTargetPositions(const Map::mapType& _map, const Point _end,
                                         const int _carRadius, const int _fireRadius, SubDivLevel searchLevel) {
        std::vector<Point> targets;
        const int minDist = _fireRadius;
        const int maxDist = 2 * _fireRadius;

        // 分析障碍物并找出清晰扇区
        auto clearSectors = findSectors(_map, _end, searchLevel,2.0*_fireRadius);

        std::cout << "DEBUG: Found " << clearSectors.size() << " clear sectors\n";

        for (const auto& sector : clearSectors) {
            std::cout << "DEBUG: Sector from " << sector.start << "° to " << sector.end << "°\n";
        }

        if (clearSectors.empty()) {
            std::cout << "DEBUG: No clear sectors found\n";
            return {};
        }

        // 在目标点周围搜索有效位置
        const int searchRange = maxDist + _carRadius;

        for (int dy = -searchRange; dy <= searchRange; ++dy) {
            for (int dx = -searchRange; dx <= searchRange; ++dx) {
                Point targetPos = {_end.x + dx, _end.y + dy};

                // 检查是否在地图边界内
                if (targetPos.x >= 0 && targetPos.x < static_cast<int>(_map[0].size()) &&
                    targetPos.y >= 0 && targetPos.y < static_cast<int>(_map.size())) {

                    const double dist = std::sqrt(dx * dx + dy * dy);

                    // 检查距离约束：必须在环形区域内且在禁入区外
                    if (dist >= minDist && dist <= maxDist && dist > _fireRadius &&
                        isValidPosition(_map, targetPos, _end, _carRadius, _fireRadius)) {

                        // 检查是否在清晰扇区内
                        if (isPositionInClearSector(targetPos, _end, clearSectors)) {
                            targets.push_back(targetPos);
                        }
                    }
                }
            }
        }

        return targets;
    }

    // 启发式函数（欧几里得距离）
    double heuristic(Point pos1, Point pos2) {
        return std::sqrt((pos1.x - pos2.x) * (pos1.x - pos2.x) +
                        (pos1.y - pos2.y) * (pos1.y - pos2.y));
    }

    // 获取邻居节点
    std::vector<Point> getNeighbors(Point pos, const Map::mapType& _map, Point start, Point end,
                                   int carRadius, int fireRadius, SubDivLevel searchLevel) {
        std::vector<Point> neighbors;
        const double stepSize = getStepSizeForLevel(searchLevel);
        const int intStepSize = static_cast<int>(std::round(stepSize));

        // 根据步进大小调整方向向量
        const std::vector<std::pair<int, int>> directions = {
            {0, intStepSize}, {intStepSize, 0}, {0, -intStepSize}, {-intStepSize, 0},    // 上下左右
            {intStepSize, intStepSize}, {intStepSize, -intStepSize},
            {-intStepSize, intStepSize}, {-intStepSize, -intStepSize}   // 对角线
        };

        auto clearSectors = findSectors(_map, end, searchLevel,fireRadius);

        for (const auto& dir : directions) {
            Point newPos = {pos.x + dir.first, pos.y + dir.second};

            if (isValidPosition(_map, newPos, end, carRadius, fireRadius)) {
                // 额外检查：位置必须在清晰扇区内或者是起始位置
                if (newPos.x == start.x && newPos.y == start.y ||
                    isPositionInClearSector(newPos, end, clearSectors)) {
                    neighbors.push_back(newPos);
                }
            }
        }

        return neighbors;
    }

    // A*路径搜索算法
    std::vector<Point> aStarSearch(Point start, Point target, const Map::mapType& _map,
                                  Point end, double carRadius, double fireRadius, SubDivLevel searchLevel) {

        // 使用priority_queue实现A*算法
        auto cmp = [](const std::pair<double, Point>& a, const std::pair<double, Point>& b) {
            return a.first > b.first;  // 最小堆
        };
        std::priority_queue<std::pair<double, Point>,
                           std::vector<std::pair<double, Point>>,
                           decltype(cmp)> openSet(cmp);

        std::unordered_map<long long, Point> cameFrom;
        std::unordered_map<long long, double> gScore;
        std::unordered_set<long long> closedSet;

        // 将Point转换为唯一的long long键
        auto pointToKey = [](Point p) -> long long {
            return (static_cast<long long>(p.x) << 32) | static_cast<long long>(p.y);
        };

        auto keyToPoint = [](long long key) -> Point {
            return {static_cast<int>(key >> 32), static_cast<int>(key & 0xFFFFFFFF)};
        };

        long long startKey = pointToKey(start);
        long long targetKey = pointToKey(target);

        openSet.push({0.0, start});
        gScore[startKey] = 0.0;

        auto clearSectors = findSectors(_map, end, searchLevel,fireRadius);
        const int minDist = fireRadius;
        const int maxDist = 2 * fireRadius;

        std::cout << "DEBUG: A* starting with " << clearSectors.size() << " clear sectors (level " << searchLevel << ")\n";

        int stepsCount = 0;
        while (!openSet.empty()) {
            stepsCount++;
            Point current = openSet.top().second;
            openSet.pop();

            long long currentKey = pointToKey(current);

            if (closedSet.count(currentKey)) {
                continue;
            }
            closedSet.insert(currentKey);

            // 积极早期终止：如果当前位置在清晰扇区内且满足距离约束
            if (!(current.x == start.x && current.y == start.y)) {
                const double distToEnd = std::sqrt((current.x - end.x) * (current.x - end.x) +
                                                  (current.y - end.y) * (current.y - end.y));

                if (distToEnd >= minDist && distToEnd <= maxDist && distToEnd > fireRadius &&
                    isPositionInClearSector(current, end, clearSectors)) {

                    std::cout << "DEBUG: Early termination at (" << current.x << ", " << current.y
                             << ") after " << stepsCount << " steps\n";

                    // 重构路径
                    std::vector<Point> path;
                    Point tempCurrent = current;
                    long long tempKey = currentKey;
                    while (cameFrom.count(tempKey)) {
                        path.push_back(tempCurrent);
                        tempCurrent = cameFrom[tempKey];
                        tempKey = pointToKey(tempCurrent);
                    }
                    path.push_back(start);
                    std::reverse(path.begin(), path.end());
                    return path;
                }
            }

            // 正常终止：到达原始目标
            if (currentKey == targetKey) {
                std::cout << "DEBUG: Reached original target\n";
                std::vector<Point> path;
                Point tempCurrent = current;
                long long tempKey = currentKey;
                while (cameFrom.count(tempKey)) {
                    path.push_back(tempCurrent);
                    tempCurrent = cameFrom[tempKey];
                    tempKey = pointToKey(tempCurrent);
                }
                path.push_back(start);
                std::reverse(path.begin(), path.end());
                return path;
            }

            // 探索邻居节点
            auto neighbors = getNeighbors(current, _map, start, end, carRadius, fireRadius, searchLevel);
            for (const auto& neighbor : neighbors) {
                long long neighborKey = pointToKey(neighbor);

                if (closedSet.count(neighborKey)) {
                    continue;
                }

                // 计算移动成本
                const double dx = std::abs(neighbor.x - current.x);
                const double dy = std::abs(neighbor.y - current.y);
                const double moveCost = std::sqrt(dx * dx + dy * dy);

                const double tentativeGScore = gScore[currentKey] + moveCost;

                if (!gScore.count(neighborKey) || tentativeGScore < gScore[neighborKey]) {
                    cameFrom[neighborKey] = current;
                    gScore[neighborKey] = tentativeGScore;
                    const double fScore = tentativeGScore + heuristic(neighbor, target);

                    openSet.push({fScore, neighbor});
                }
            }
        }

        return {};  // 未找到路径
    }

    // 将路径转换为Movement对象
    std::vector<Movement> pathToMovements(const std::vector<Point>& path,
                                         const Map::mapType& _map, Point end,
                                         int carRadius, int fireRadius) {
        if (path.size() < 2) {
            return {};
        }

        // 检查是否可以直接移动
        auto canMoveDirectly = [&](Point startPos, Point endPos) -> bool {
            const int dx = std::abs(endPos.x - startPos.x);
            const int dy = std::abs(endPos.y - startPos.y);

            if (dx == 0 && dy == 0) {
                return true;
            }

            const int steps = std::max(dx, dy);

            // 检查路径上的每个点
            for (int i = 0; i <= steps; ++i) {
                const double t = static_cast<double>(i) / std::max(steps, 1);
                const Point checkPos = {
                    static_cast<int>(startPos.x + t * (endPos.x - startPos.x)),
                    static_cast<int>(startPos.y + t * (endPos.y - startPos.y))
                };

                if (!isValidPosition(_map, checkPos, end, carRadius, fireRadius)) {
                    return false;
                }
            }
            return true;
        };

        // 首先简化路径，连接远距离点
        std::vector<Point> simplifiedPath = {path[0]};
        size_t i = 0;

        while (i < path.size() - 1) {
            Point currentPos = path[i];

            // 找到可以直接到达的最远点
            size_t maxReachable = i + 1;
            for (size_t j = i + 2; j < path.size(); ++j) {
                if (canMoveDirectly(currentPos, path[j])) {
                    maxReachable = j;
                } else {
                    break;
                }
            }

            simplifiedPath.push_back(path[maxReachable]);
            i = maxReachable;
        }

        std::cout << "DEBUG: Simplified path from " << path.size() << " to "
                 << simplifiedPath.size() << " points\n";

        // 将简化后的路径转换为Movement对象
        std::vector<Movement> movements;

        for (size_t i = 0; i < simplifiedPath.size() - 1; ++i) {
            Point currentPos = simplifiedPath[i];
            Point nextPos = simplifiedPath[i + 1];

            // 计算移动向量
            const double dx = nextPos.x - currentPos.x;
            const double dy = nextPos.y - currentPos.y;

            // 计算距离
            const double distance = std::sqrt(dx * dx + dy * dy);

            // 计算角度（度，0° = 东，90° = 北）
            double angle = std::atan2(dy, dx) * 180.0 / M_PI;

            // 归一化角度到[0, 360)范围
            if (angle < 0) {
                angle += 360.0;
            }

            movements.push_back({angle, distance});

            std::cout << "DEBUG: Movement " << (i + 1) << ": angle=" << angle
                     << "°, distance=" << distance << "\n";
        }

        return movements;
    }

    // 主要的路径搜索函数
    std::vector<Movement> SearchPathExport(const Map::mapType& _map, Point start, Point end,
                                    int carRadius, int fireRadius, SubDivLevel searchLevel) {
        if (_map.empty() || _map[0].empty()) {
            return {};
        }

        if (!isValidPosition(_map, start, end, carRadius, fireRadius)) {
            std::cout << "DEBUG: Invalid start position!\n";
            return {};
        }

        std::cout << "DEBUG: Starting search with level " << searchLevel << " (stepSize: " << getStepSizeForLevel(searchLevel) << ")\n";

        // 获取有效的目标位置
        std::cout << "DEBUG: Getting target positions...\n";
        auto targetPositions = getTargetPositions(_map, end, carRadius, fireRadius, searchLevel);
        std::cout << "DEBUG: Found " << targetPositions.size() << " valid target positions\n";

        if (targetPositions.empty()) {
            std::cout << "DEBUG: No valid target positions found!\n";
            return {};
        }

        // 找到离起始点最近的有效目标位置
        Point bestTarget = *std::min_element(targetPositions.begin(), targetPositions.end(),
            [&](const Point& a, const Point& b) {
                return heuristic(start, a) < heuristic(start, b);
            });

        std::cout << "DEBUG: Selected best target: (" << bestTarget.x << ", " << bestTarget.y << ")\n";

        // 使用A*算法找到路径
        std::cout << "DEBUG: Starting A* pathfinding...\n";
        auto path = aStarSearch(start, bestTarget, _map, end, carRadius, fireRadius, searchLevel);
        if (path.empty()) {
            std::cout << "DEBUG: A* failed to find path!\n";
            return {};
        }

        std::cout << "DEBUG: A* found path with " << path.size() << " points\n";

        // 将路径转换为Movement对象
        auto movements = pathToMovements(path, _map, end, carRadius, fireRadius);
        std::cout << "DEBUG: Generated " << movements.size() << " movement commands\n";

        return movements;
    }

}  // namespace Path