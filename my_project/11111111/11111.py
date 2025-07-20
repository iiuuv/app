import subprocess
import mmap
import os

def create_shared_memory(data, size=1024):
    # 创建临时文件用于共享内存
    temp_file = '/tmp/shared_memory'
    with open(temp_file, 'wb') as f:
        # 调整文件大小
        f.write(b'\0' * size)
        # 写入数据
        f.seek(0)
        f.write(data.encode())

    return temp_file

def main():
    # 要共享的数据
    shared_data = "Hello from Python!"
    # 创建共享内存
    shm_name = create_shared_memory(shared_data)

    print(f"共享内存已创建: {shm_name}")
    print(f"数据内容: {shared_data}")

    try:
        # 使用subprocess启动C++程序，并传递共享内存名称作为参数
        print("正在启动C++程序...")
        result = subprocess.run(
            ["/app/my_project/11111111/1111111", shm_name],
            capture_output=True,
            text=True,
            check=True
        )

        print("C++程序输出:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error: C++程序执行失败: {e.stderr}")
    finally:
        # 清理临时文件
        os.remove(shm_name)
        print("共享内存文件已删除")

if __name__ == "__main__":
    main()