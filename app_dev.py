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
#from apps.esp.esp_main import EspMain as EspMain
from apps.rgl.rgl_main import RglMain as RglMain

def test1():
    # 试验读取静态方法
    #CQues.test()
    db.init_db_pool()
    CRecommendEngine.test()
    
def test_esp():
    #EspMain.startup({})
    RglMain.startup({})
    
if __name__ == '__main__':
    print('starting up...')
    #test1()
    test_esp()
    
