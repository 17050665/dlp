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
from jinja2 import Template
import jinja2 as jinja2
#
from controller.c_ques import CQues as CQues

#params = {}
#lcct.startup(params)
#aiw_main.startup(params)
#mcv_main.startup(params)
#fai_main.startup(params)

def read_html(file):
    fo = open(ag.upload_dir + file, 'r', encoding='utf-8')
    try:
        html = fo.read()
    finally:
        fo.close()
    return html

def test1():
    stut_id = 2
    ques_seq = 1
    html_str = ''
    # 获取当前的练习编号
    excs_id = excs.get_excs_id(stut_id)
    ques_num = excs.get_excs_ques_num(excs_id)
    # 获取练习下面题目
    ques_id, ques_type_id = excs.get_excs_ques(excs_id, ques_seq)
    filePath = excs.get_ques_stem_file(ques_id)    
    templateLoader = jinja2.FileSystemLoader( searchpath='D:/')
    templateEnv = jinja2.Environment( loader=templateLoader )
    ques_stem_file = ag.upload_dir + filePath
    template = templateEnv.get_template( ques_stem_file )
    html_str = template.render( {'ques_num': ques_num, 'ques_seq': ques_seq} )
    html_str += '\r\n<div class="weui-cells weui-cells_radio">'
    optns = excs.get_ques_optns(ques_id)
    for optn in optns:
        optn_tpl = templateEnv.get_template(ag.upload_dir + optn[1])
        stut_ques_optn_id = excs.get_stut_ss_ques_ansr(stut_id, excs_id, ques_id)
        if stut_ques_optn_id == optn[0]:
            check_status = 'checked=true'
        else:
            check_status = ''
        html_str += optn_tpl.render({'optn_page_id': 'o_{0}_{1}_{2}_{3}'.format(stut_id, excs_id, ques_id, optn[0]), 'checked_status': check_status})
    html_str += '\r\n</div>'
    # 根据问题类型取相应的javascript程序
    
    
    
    print(html_str)

if __name__ == '__main__':
    print('starting up...')
    db.init_db_pool()
    print('step 1')
    
    stut_id = 2
    excs_id = 1
    ques_id = 1
    #
    #print('num={0}'.format(num))
    # test1()
    # 试验读取静态方法
    obj = CQues()
    m1 = getattr(CQues, 'get_ques_html')
    params = {'args': (), 'kwargs': {'stut_id': 2, 'ques_seq': 1}}
    resp = m1(params)
    print(resp['html'])
    
    
    
    '''
    html = ''
    rows = ex.get_stut_ques(1)
    for row in rows:
        ques_id = row[0]
        recs = ex.get_ques_stem(ques_id)
        filePath = recs[0][0]
        html = read_html(filePath)
        html += '<div class="weui-cells weui-cells_radio">'
        optns = ex.get_ques_optns(ques_id)
        for optn in optns:
            html += read_html(optn[1])
        html += '</div>'
    print(html)
    sys.exit()
    '''
    
    '''
    wqs.init_wky_queues()
    app_web.startup()
    ag.rdb_pool_cleaner.join()
    ag.wdb_pool_cleaner.join()
    ag.task_q_thread.join()
    ag.result_q_thread.join()
    '''
