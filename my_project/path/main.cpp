#include <vector>
#include <optional>
#include <iostream>
#include "json.hpp"
#include "path.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sys/eventfd.h>


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
#define SHM_MAP "/dev/shm/shm_map"
#define SHM_DATA "/dev/shm/shm_json"
#define SHM_RESULT "/dev/shm/shm_result"
// 数据大小
#define MAP_SIZE (133 * 100)
#define DATA_SIZE 200        // 4 个 int（修正为 4 个）
#define RESULT_SIZE 2048

int main() {

    int map_fd = open(SHM_MAP, O_RDONLY);
    void* map_ptr = mmap(nullptr, MAP_SIZE, PROT_READ, MAP_SHARED, map_fd, 0);
    bool * pyMap = static_cast<bool*>(map_ptr);

    int data_fd = open(SHM_DATA, O_RDONLY);
    void* data_ptr = mmap(nullptr, DATA_SIZE, PROT_READ, MAP_SHARED, data_fd, 0);
    char * pyArgs = static_cast<char*>(data_ptr);
    
    int result_fd = open(SHM_RESULT, O_RDWR);
    void* result_ptr = mmap(nullptr, RESULT_SIZE, PROT_WRITE, MAP_SHARED, result_fd, 0);
    char * pyReturn = static_cast<char*>(result_ptr);

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

    std::cout << "=== Path Finding ===" << std::endl;
    std::cout << "Start: (" << start.x << ", " << start.y << ") -> Target: (" << end.x << ", " << end.y << ")" << std::endl;
    std::cout << "Car Radius: " << carRadius << ", Fire Radius: " << fireRadius << std::endl;


    auto movements = Path::SearchPathExport(map, start, end, carRadius, fireRadius, Path::SubDivLevel::minSize);

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

    std::snprintf(pyReturn, 2048, "%s", result.dump().c_str());

    std::cout << std::endl;
    return 0;
}
