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
    obj = CQues()
    mtd = getattr(CQues, 'test')
    mtd()
    '''
    json_obj = {"stut_id":"2","excs_id":"1","ques_id":"1","optn_id":"2"}
    params = {'args': (), 'kwargs': {'json_obj': json_obj}}
    resp = m1(params)
    print(resp)
    '''

if __name__ == '__main__':
    print('starting up...')
    db.init_db_pool()
    test1()
    
    