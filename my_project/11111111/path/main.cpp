#include <vector>
#include <optional>
#include <iostream>
#include "json.hpp"
#include "path.hpp"

#define M_PI 3.14159265358979323846

/*
    {
        "width": 200,
        "height": 200,
        "start_point": {
            "x": 10,
            "y": 10
        },
        "end_point": {
            "x": 160,
            "y": 160
        },
        "car_radius": 8,
        "fire_radius": 8,
    }


    [
        {
            "Direction" : 60.0,
            "Distance" : 10.0
        },
        {
            "Direction" : 60.0,
            "Distance" : 10.0
        },
    ]
*/

using json = nlohmann ::json;

int main() {

    bool * pyMap = nullptr;
    char * pyArgs = nullptr;
    char * pyReturn = nullptr;

    char * args = pyArgs;
    json argsJson = json::parse(args);


    // 测试完整的路径搜索算法
    const Path::Point start = {argsJson["start_point"]["x"], argsJson["start_point"]["y"]};    // 起始位置
    const Path::Point end = {argsJson["end_point"]["x"], argsJson["end_point"]["y"]};        // 目标位置
    const int carRadius = argsJson["car_radius"];       // 车辆半径
    const int fireRadius = argsJson["fire_radius"];           // 射击半径

    unsigned int width = argsJson["width"];
    unsigned int height = argsJson["height"];
    std::vector<std::vector<bool>> map(height, std::vector<bool>(width, false));

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            map[y][x] = pyMap[y * width + x] != 0;
        }
    }

    std::cout << "=== 路径搜索测试 ===" << std::endl;
    std::cout << "起始位置: (" << start.x << ", " << start.y << ")" << std::endl;
    std::cout << "目标位置: (" << end.x << ", " << end.y << ")" << std::endl;
    std::cout << "车辆半径: " << carRadius << std::endl;
    std::cout << "射击半径: " << fireRadius << std::endl;
    std::cout << std::endl;

    auto movements = Path::SearchPathExport(map, start, end, carRadius, fireRadius, Path::SubDivLevel::maxSize);

    // 执行路径搜索
    json result;
    if (movements.empty()) {
        result["Error"] = "No available path found.";
    }
    else {
        for (unsigned int i = 0; i < movements.size(); ++i) {
            result[i] = {
                {"Direction", movements[i].angle},
                {"Distance", movements[i].distance}
            };
        }
    }

    std::snprintf(pyReturn, 1024, "%s", result.dump().c_str());

    std::cout << std::endl;
    return 0;
}
