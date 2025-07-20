#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sys/eventfd.h>

// 共享内存名称
#define SHM_MAP "/dev/shm/shm_map"
#define SHM_DATA "/dev/shm/shm_json"
#define SHM_RESULT "/dev/shm/shm_result"

// 数据大小
#define MAP_SIZE (133 * 100)
#define DATA_SIZE 4096        // 4 个 int（修正为 4 个）
#define RESULT_SIZE (4 * 4)

int main(int argc, char* argv[]) {
    std::ofstream log("/tmp/cpp_output.log", std::ios::app);
    if (!log) {
        std::cerr << "无法打开日志文件" << std::endl;
        return -1;
    }

    if (argc < 2) {
        log << "错误：缺少 eventfd 参数" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <eventfd>" << std::endl;
        return -1;
    }

    // Step 1: 获取 eventfd 文件描述符
    int efd = atoi(argv[1]);
    log << "C++ 已启动，等待通知..." << std::endl;

    // Step 2: 阻塞等待通知
    eventfd_t value;
    eventfd_read(efd, &value);
    log << "收到通知，value = " << value << std::endl;

    // Step 3: 映射地图共享内存
    int map_fd = open(SHM_MAP, O_RDONLY);
    void* map_ptr = mmap(nullptr, MAP_SIZE, PROT_READ, MAP_SHARED, map_fd, 0);
    const uint8_t* map_data = static_cast<const uint8_t*>(map_ptr);

    // Step 4: 映射坐标共享内存
    int data_fd = open(SHM_DATA, O_RDONLY);
    void* data_ptr = mmap(nullptr, DATA_SIZE, PROT_READ, MAP_SHARED, data_fd, 0);
    const int32_t* coords = static_cast<const int32_t*>(data_ptr);

    // Step 5: 映射回传共享内存
    int result_fd = open(SHM_RESULT, O_RDWR);
    void* result_ptr = mmap(nullptr, RESULT_SIZE, PROT_WRITE, MAP_SHARED, result_fd, 0);
    int32_t* result = static_cast<int32_t*>(result_ptr);

    // Step 6: 打印部分地图数据
    log << "地图数据前 10x10：" << std::endl;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            log << static_cast<int>(map_data[i * 100 + j]) << " ";
        }
        log << std::endl;
    }

    // Step 7: 打印坐标数据
    log << "收到坐标数据: (" 
        << coords[0] << ", " << coords[1] << ") -> ("
        << coords[2] << ", " << coords[3] << ")" << std::endl;

    // Step 8: 模拟处理后回传数据
    result[0] = coords[0] + 1;
    result[1] = coords[1] + 1;
    result[2] = coords[2] + 1;
    result[3] = coords[3] + 1;

    log << "回传数据: [" 
        << result[0] << ", " << result[1] 
        << ", " << result[2] << ", " << result[3] << "]" << std::endl;

    // Step 9: 清理资源
    munmap(map_ptr, MAP_SIZE);
    munmap(data_ptr, DATA_SIZE);
    munmap(result_ptr, RESULT_SIZE);
    close(map_fd);
    close(data_fd);
    close(result_fd);
    close(efd);

    return 0;
}

// #include <iostream>
// #include <fstream>
// #include <sys/eventfd.h>
// #include <unistd.h>
// #include <cerrno>
// #include <cstring>

// int main(int argc, char* argv[]) {
//     std::cout << "[DEBUG] C++ 程序启动" << std::endl;

//     if (argc < 2) {
//         std::cerr << "[ERROR] 缺少 eventfd 参数" << std::endl;
//         return -1;
//     }

//     int efd = atoi(argv[1]);
//     std::cout << "[DEBUG] eventfd = " << efd << std::endl;

//     std::ofstream log("/tmp/cpp_output.log", std::ios::app);
//     if (!log) {
//         std::cerr << "[ERROR] 无法打开日志文件" << std::endl;
//         return -1;
//     }

//     log << "C++ 程序启动" << std::endl;

//     std::cout << "[DEBUG] 等待通知..." << std::endl;
//     eventfd_t value;
//     int ret = eventfd_read(efd, &value);
//     if (ret != 0) {
//         log << "[ERROR] eventfd_read 失败，errno = " << errno << std::endl;
//         std::cerr << "[ERROR] eventfd_read 失败，errno = " << errno << std::endl;
//         return -1;
//     }

//     std::cout << "[DEBUG] 收到通知，value = " << value << std::endl;
//     log << "收到通知，value = " << value << std::endl;

//     // 后续处理逻辑
//     ...
// }