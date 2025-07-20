#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc != 4) {
        std::cerr << "用法: " << argv[0] << " <共享内存名称>" << std::endl;
        return 1;
    }

    // 从命令行参数获取共享内存名称
    const char* shm_name = argv[1];



    // 打开共享内存文件
    int fd = open(shm_name, O_RDONLY);



    if (fd == -1) {
        perror("无法打开共享内存文件");
        return 1;
    }

    // 获取文件大小
    off_t size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);



    // 创建内存映射
    void* addr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);

    
    if (addr == MAP_FAILED) {
        perror("内存映射失败");
        close(fd);
        return 1;
    }

    // 读取共享内存内容
    std::cout << "从共享内存读取的数据: " << static_cast<char*>(addr) << std::endl;

    // 清理资源
    if (munmap(addr, size) == -1) {
        perror("解除内存映射失败");
    }
    close(fd);

    return 0;
}