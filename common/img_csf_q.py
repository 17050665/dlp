import threading
import queue
import app_global as ag
import apps.lcct.slim_csf_inresv2 as slim_csf_inresv2
#import controller.c_mlp as c_mlp

class Img_Csf_Q_Thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        print('初始化任务队列处理线程')

    def run(self):
        print('启动图像识别处理线程')
        sci = slim_csf_inresv2.SlimCsfInResV2()
        sci.startup()
        


