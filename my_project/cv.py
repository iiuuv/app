from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

def ftp_server():
    # 创建用户授权管理器
    authorizer = DummyAuthorizer()
    
    # 添加用户权限（将图片目录添加为访问路径）
    # authorizer.add_user("user", "12345", "/home/user", perm="elradfmw")  # 原目录
    authorizer.add_user("user", "12345", "/app/my_project/photo", perm="elradfmw")  # 新增图片目录
    
    # 可选：添加匿名用户访问（允许无账号访问）
    # authorizer.add_anonymous("/home/nobody")
    authorizer.add_anonymous("/app/my_project/photo")  # 允许匿名用户访问图片目录

    # 初始化FTP处理程序
    handler = FTPHandler
    handler.authorizer = authorizer

    # 设置服务器
    server = FTPServer(("192.168.1.38", 8080), handler)
    
    # 启动服务器
    try:
        print("FTP服务器已启动，等待连接...")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已关闭")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    ftp_server()