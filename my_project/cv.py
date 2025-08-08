import threading
import queue
import time

# 创建线程安全的队列
msg_queue = queue.Queue(maxsize=5)  # 最大容量5

def sender():
    for i in range(3):
        time.sleep(1)
        message = f"消息 {i}"
        msg_queue.put(message)  # 放入队列
        print(f"发送: {message}")

def receiver():
    while True:
        message = msg_queue.get()  # 从队列取出（若为空则阻塞等待）
        print(f"接收: {message}")
        msg_queue.task_done()  # 通知队列任务完成

t1 = threading.Thread(target=sender)
t2 = threading.Thread(target=receiver, daemon=True)

t1.start()
t2.start()

t1.join()
msg_queue.join()  # 等待队列中所有消息被处理完
print("程序结束")