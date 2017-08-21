import sys
sys.path.append('./lib/cherrypy')
sys.path.append('./lib/jinja')
import os
import cherrypy
import app_global as ag
import model.m_mysql as db
import app_web as app_web
import common.wky_queues as wqs
#
from controller.c_ques import CQues as CQues


def test1():
    # 试验读取静态方法
    CQues.test()

if __name__ == '__main__':
    print('starting up...')
    db.init_db_pool()
    test1()
    
    