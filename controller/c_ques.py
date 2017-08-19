import sys
sys.path.append('./lib/cherrypy')
sys.path.append('./lib/jinja')
import cherrypy
import json
from jinja2 import Template
import jinja2 as jinja2
import conf.web_conf as web_conf
import app_global as ag
import model.mf_excs as excs

class CQues(object):
    exposed = True
    def __init__(self):
        self.web_dir = ag.web_dir
        
    @staticmethod
    def get_ques_html(req_args):
        ''' 获取题目的HTML内容 '''
        params = req_args['kwargs']
        print('生成问题页面HTML内容:{0}'.format(params))
        stut_id = params['stut_id']
        ques_seq = params['ques_seq']
        if 'excs_id' in params:
            excs_id = params['excs_id']
        else:
            excs_id = excs.get_excs_id(stut_id)
        ques_num = excs.get_excs_ques_num(excs_id)
        ques_id, ques_type_id = excs.get_excs_ques(excs_id, ques_seq)
        
        tpl_loader = jinja2.FileSystemLoader(searchpath='/')
        tpl_env = jinja2.Environment(loader=tpl_loader)
        ques_stem_file = '{0}tpl/qs_{1}.html'.format(ag.resources_dir, ques_type_id)
        ques_stem_tpl = tpl_env.get_template(ques_stem_file)
        ques_stem_dict = excs.get_ques_stem_param(ques_id)
        html_str = ques_stem_tpl.render({'ques_num': ques_num, 'ques_seq': ques_seq, 'title': ques_stem_dict['title']})
        html_str += '<div class="weui-cells weui-cells_radio">'
        optns = excs.get_ques_optns(ques_id)
        for optn in optns:
            optn_tpl = tpl_env.get_template(ag.resources_dir + optn[1])
            stut_ques_optn_id = excs.get_stut_ss_ques_ansr(stut_id, excs_id, ques_id)
            optn_param_dict = excs.get_ques_optn_param(optn[0])
            if stut_ques_optn_id == optn[0]:
                check_status = 'checked=true'
            else:
                check_status = ''
            html_str += optn_tpl.render({'optn_id': 'o_{0}_{1}_{2}_{3}'.format(stut_id, excs_id, ques_id, optn[0]), 'optn_text': optn_param_dict['optn_text'], 'checked_status': check_status, 'optn_x_id': optn[0]})
        html_str += '</div>'
        fo = open('{0}tpl/qs_{1}.js'.format(ag.resources_dir, ques_type_id), 'r', encoding='utf-8')
        try:
            js = fo.read()
        finally:
            fo.close()
        html_str += js
    
        resp = {}
        resp['status'] = 'Ok'
        resp['excs_id'] = excs_id
        resp['ques_num'] = ques_num
        resp['html'] = html_str
        return resp

