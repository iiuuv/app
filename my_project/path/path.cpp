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

    std::vector<Sector> findSectors(const Map::mapType& _map, const Point _endPoint, SubDivLevel searchLevel, double fireRadius, double carRadius) {
        if (_map.empty() || _map[0].empty()) {
            return {};
        }

        const int height = static_cast<int>(_map.size());
        const int width = static_cast<int>(_map[0].size());
        const double stepSize = getRayStepSizeForLevel(searchLevel);

        // 射线检测半径为 fireRadius + 2*carRadius
        const double rayLength = fireRadius + 2 * carRadius;

        // 计算目标点到地图边界的最小距离
        double minDistToBorder = std::min({
            static_cast<double>(_endPoint.x),
            static_cast<double>(_endPoint.y),
            static_cast<double>(width - _endPoint.x - 1),
            static_cast<double>(height - _endPoint.y - 1)
        });

        // 如果目标点太靠近边界，适当缩短射线长度
        double effectiveRayLength = std::min(rayLength, minDistToBorder);

        std::vector blockedAngles(360, false);

        // 向360度各个方向投射射线，检查是否被阻挡
        int blockedCount = 0;
        for (int angleDeg = 0; angleDeg < 360; ++angleDeg) {
            const double angleRad = angleDeg * M_PI / 180.0;
            const double dx = std::cos(angleRad);
            const double dy = std::sin(angleRad);

            bool isBlocked = false;

            // 沿射线方向逐步检测到指定距离
            for (int step = 1; step <= static_cast<int>(effectiveRayLength / stepSize); ++step) {
                const double rayDistance = step * stepSize;
                const double rayX = _endPoint.x + dx * rayDistance;
                const double rayY = _endPoint.y + dy * rayDistance;

                // 检查是否撞到障碍物
                const int gridX = static_cast<int>(rayX);
                const int gridY = static_cast<int>(rayY);

                if (gridX >= 0 && gridY >= 0 && gridX < width && gridY < height) {
                    if (_map[gridY][gridX]) {  // 撞到障碍物
                        isBlocked = true;
                        break;
                    }
                }
            }

            blockedAngles[angleDeg] = isBlocked;
            if (isBlocked) blockedCount++;
        }

        std::cout << "Ray casting: " << (360 - blockedCount) << "/360 directions clear\n";

        // 找出连续的清晰角度区域，要求至少45度
        std::vector<Sector> tempSectors;
        const int minGapDegrees = 45;

        // 简化逻辑：直接扫描所有角度，找出连续的清晰区域
        std::vector<std::pair<int, int>> gaps;

        int start = -1;
        for (int angle = 0; angle < 360; ++angle) {
            if (!blockedAngles[angle]) {
                // 清晰角度
                if (start == -1) {
                    start = angle;
                }
            } else {
                // 阻挡角度
                if (start != -1) {
                    gaps.push_back({start, angle});
                    start = -1;
                }
            }
        }

        // 处理最后一个可能的清晰区域（到360度）
        if (start != -1) {
            gaps.push_back({start, 360});
        }

        // 检查是否有跨越0度的情况（第一个和最后一个区域可能连接）
        if (!gaps.empty() && !blockedAngles[0] && !blockedAngles[359]) {
            // 合并第一个和最后一个区域
            if (gaps.size() >= 2 && gaps[0].first == 0 && gaps.back().second == 360) {
                int start = gaps.back().first;
                int end = gaps[0].second;
                int gapSize = (360 - start) + end;
                if (gapSize >= minGapDegrees) {
                    // 跨越0度的扇区，start > end
                    tempSectors.push_back({static_cast<double>(start), static_cast<double>(end)});
                }
                // 移除已合并的区域
                gaps.erase(gaps.begin());
                gaps.pop_back();
            }
        }
        // 添加剩余的正常区域
        for (const auto& gap : gaps) {
            int gapSize = gap.second - gap.first;
            if (gapSize >= minGapDegrees) {
                tempSectors.push_back({static_cast<double>(gap.first), static_cast<double>(gap.second)});
            }
        }
        // 扇区归一化到[0,360)
        for (auto& sector : tempSectors) {
            if (sector.start >= 360.0) sector.start -= 360.0;
            if (sector.end >= 360.0) sector.end -= 360.0;
        }

        std::vector<Sector> clearSectors;

        for (auto sector : tempSectors) {
            if (sector.start == 0 && sector.end == 0.0) {
                // 特例：如果扇区是完整的360度，直接返回
                clearSectors.push_back({0.0, 360.0});
                continue;
            }
            if (sector.start < sector.end) {
                // 正常扇区
                clearSectors.push_back(sector);
            }
            else {
                // 跨越0度的扇区
                clearSectors.push_back({0, sector.start});
                clearSectors.push_back({sector.end, 360});
            }

        }

        std::cout << "Found " << clearSectors.size() << " sectors\n";

        for (const auto& sector : clearSectors) {
            std::cout << "Sector: [" << sector.start << ", " << sector.end << "]\n";
        }

        return clearSectors;
    }



    // 检查位置是否有效（考虑车辆半径和禁入区域）
    bool isValidPosition(const Map::mapType& _map,const Point _start,const Point _end,const int _carRadius,const int _fireRadius) {
        const int height = static_cast<int>(_map.size());
        const int width = static_cast<int>(_map[0].size());

        // 检查禁入区域：车辆不能进入距离目标点小于fireRadius的区域
        const double distToEnd = std::sqrt((_start.x - _end.x) * (_start.x - _end.x) +
                                          (_start.y - _end.y) * (_start.y - _end.y));
        if (distToEnd < _fireRadius) {
            return false;
        }

        if (_start.x > width || _start.y > height ||
            _start.x < 0 || _start.y < 0) {
            return false; // 超出地图边界
        }

        // 检查车辆半径范围内的圆形区域是否有障碍物
        for (int dy = -_carRadius; dy <= _carRadius; ++dy) {
            for (int dx = -_carRadius; dx <= _carRadius; ++dx) {
                if (dx * dx + dy * dy <= _carRadius * _carRadius) {  // 圆形碰撞检测
                    const int checkX = _start.x + dx;
                    if (const int checkY = _start.y + dy; checkX >= 0 && checkX < width && checkY >= 0 && checkY < height) {
                        if (_map[checkY][checkX] == true) {  // 撞到障碍物
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

        // 目标点应该在 fireRadius + carRadius 的圆上
        const double targetDistance = _fireRadius + _carRadius;

        auto clearSectors = findSectors(_map, _end, searchLevel, _fireRadius, _carRadius);

        if (clearSectors.empty()) {
            std::cout << "❌ No available sectors: target surrounded by obstacles\n";
            return {};
        }

        // 在清晰扇区内，沿着 targetDistance 圆周寻找目标点
            for (const auto&[start, end] : clearSectors) {
            // 在每个扇区内每隔2度寻找一个目标点
            for (double angle = start; angle < end; angle += 1.0) {
                const double angleRad = angle * M_PI / 180.0;

                const int targetX = static_cast<int>(_end.x + std::cos(angleRad) * targetDistance);
                const int targetY = static_cast<int>(_end.y + std::sin(angleRad) * targetDistance);

                Point targetPos = {targetX, targetY};

                // 检查是否在地图边界内
                if (targetPos.x >= 0 && targetPos.x < static_cast<int>(_map[0].size()) &&
                    targetPos.y >= 0 && targetPos.y < static_cast<int>(_map.size())) {

                    // 验证这个位置是有效的（不在火焰半径内，且车辆可以放置）
                    if (isValidPosition(_map, targetPos, _end, _carRadius, _fireRadius)) {
                        targets.push_back(targetPos);
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

        for (const auto& dir : directions) {
            Point newPos = {pos.x + dir.first, pos.y + dir.second};

            // 只检查位置是否有效，不强制要求在清晰扇区内
            // 清晰扇区的限制应该只用于目标点选择，而不是路径搜索
            if (isValidPosition(_map, newPos, end, carRadius, fireRadius)) {
                neighbors.push_back(newPos);
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

            // 正常终止：到达目标
            if (currentKey == targetKey) {
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
            std::cout << "❌ Pathfinding failed: Invalid start position (too close to target or obstacles)\n";
            return {};
        }

        // 获取有效的目标位置
        auto targetPositions = getTargetPositions(_map, end, carRadius, fireRadius, searchLevel);

        if (targetPositions.empty()) {
            std::cout << "❌ Pathfinding failed: No reachable positions around target (surrounded by obstacles or map boundaries)\n";
            return {};
        }

        // 找到离起始点最近的有效目标位置
        Point bestTarget = *std::min_element(targetPositions.begin(), targetPositions.end(),
            [&](const Point& a, const Point& b) {
                return heuristic(start, a) < heuristic(start, b);
            });

        std::cout << "✓ Found " << targetPositions.size() << " target positions, selected best: ("
                  << bestTarget.x << ", " << bestTarget.y << ")\n";

        // 使用A*算法找到路径
        auto path = aStarSearch(start, bestTarget, _map, end, carRadius, fireRadius, searchLevel);
        if (path.empty()) {
            // 分析失败原因
            std::cout << "❌ Pathfinding failed: A* algorithm cannot find path\n";

            // 检查起点到目标点之间是否有大面积障碍物阻挡
            double directDistance = heuristic(start, bestTarget);
            int obstacleCount = 0;
            int totalChecked = 0;

            // 沿直线路径检查障碍物密度
            int steps = static_cast<int>(directDistance);
            for (int i = 0; i <= steps; ++i) {
                double t = static_cast<double>(i) / std::max(steps, 1);
                int checkX = static_cast<int>(start.x + t * (bestTarget.x - start.x));
                int checkY = static_cast<int>(start.y + t * (bestTarget.y - start.y));

                if (checkX >= 0 && checkX < static_cast<int>(_map[0].size()) &&
                    checkY >= 0 && checkY < static_cast<int>(_map.size())) {
                    totalChecked++;
                    if (_map[checkY][checkX]) {
                        obstacleCount++;
                    }
                }
            }

            if (totalChecked > 0) {
                double obstacleRatio = static_cast<double>(obstacleCount) / totalChecked;
                if (obstacleRatio > 0.3) {
                    std::cout << "   Reason: High obstacle density on direct path ("
                              << static_cast<int>(obstacleRatio * 100) << "%)\n";
                } else {
                    std::cout << "   Reason: Complex terrain or large obstacles requiring detour\n";
                }
            }
            return {};
        }

        // 将路径转换为Movement对象
        auto movements = pathToMovements(path, _map, end, carRadius, fireRadius);
        std::cout << "✓ Pathfinding successful: " << movements.size() << " movement commands\n";


        auto auxAngle = std::atan2(bestTarget.y - end.y, bestTarget.x - end.x);

        auxAngle = auxAngle * 180.0 / M_PI; // 转换为度

        auxAngle += 90.0; // 调整误差角度

        if (auxAngle < 0.0) {
            auxAngle += 360.0;
        }

        movements.emplace_back(
            Movement{auxAngle, 0.0}
        );

        return movements;
    }

}  // namespace Path