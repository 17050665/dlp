from bs4 import BeautifulSoup
import requests
#from apps.rgl.spider_html_render import SpiderHtmlRender
import execjs
import json
import demjson
import csv

class RglMain(object):
    pc_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'
    pc_cookie = 'UM_distinctid=15dabfd5e91430-0c7e81214924c3-66547728-1fa400-15dabfd5e92894; qHistory=aHR0cDovL3Rvb2wuY2hpbmF6LmNvbS90b29scy9odHRwdGVzdC5hc3B4K+WcqOe6v0hUVFAgUE9TVC9HRVTmjqXlj6PmtYvor5V8aHR0cDovL3MudG9vbC5jaGluYXouY29tL3Rvb2xzL3JvYm90LmFzcHgr5pCc57Si6JyY6Jub44CB5py65Zmo5Lq65qih5ouf5oqT5Y+WfGh0dHA6Ly9zZW8uY2hpbmF6LmNvbStTRU/nu7zlkIjmn6Xor6J8aHR0cDovL3JhbmsuY2hpbmF6LmNvbSvnmb7luqbmnYPph43mn6Xor6J8aHR0cDovL3Rvb2wuY2hpbmF6LmNvbSvnq5nplb/lt6Xlhbc='
    
    post_headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        #'Cookie': pc_cookie,
        'User-Agent': pc_user_agent
    }
    
    get_headers = {
        #'Cookie': pc_cookie,
        'User-Agent': pc_user_agent
    }
    
    @staticmethod
    def get_baidu_weight(url):
        ''' 获取百度权重 '''
        url = 'http://seo.chinaz.com/{0}'.format(url)
        post_data = {
            'm': '',
            'host': 'www.gamersky.com'
        }
        wb_data = requests.post(url, headers=RglMain.post_headers, data=post_data)
        soup = BeautifulSoup(wb_data.text, 'lxml')
        tag_obj = soup.select('#seoinfo > div > ul > li:nth-of-type(2) > div.SeoMaWr01Right > div:nth-of-type(1) > p > a > img')
        tag_str = tag_obj[0].get('src')
        start_pos = tag_str.rfind('/')
        end_pos = tag_str.rfind('.gif')
        return tag_str[start_pos + 1 : end_pos]
    
    @staticmethod
    def get_all_tops(url):
        wb_data = requests.get(url, headers=RglMain.get_headers)
        html = wb_data.text
        try:
            start_pos = html.index('var rankArr = ') + 14
            end_pos = html.index('}}];', start_pos) + 3
        except Exception as ex:
            return -1, -1, -1, -1
        rankArr = html[start_pos : end_pos]
        obj = demjson.decode(rankArr)
        return obj[0]['data']['Ranked'], obj[0]['data']['alexatop'], obj[0]['data']['ClassRanked'], obj[0]['data']['ProvRanked']
        
    
    @staticmethod
    def get_alexa_ip_num(website):        
        url = 'http://alexa.chinaz.com/Handlers/GetAlexaIpNumHandler.ashx'
        post_data = {
            'url': website
        }
        wb_data = requests.post(url, headers=RglMain.post_headers, data=post_data)
        obj = demjson.decode(wb_data.text)
        if len(obj) < 1:
            return -1
        return obj[-1]['data']['IpNum']
        
    @staticmethod
    def get_alexa_pv_num(website):        
        url = 'http://alexa.chinaz.com/Handlers/GetAlexaPvNumHandler.ashx'
        post_data = {
            'url': website
        }
        wb_data = requests.post(url, headers=RglMain.post_headers, data=post_data)
        obj = demjson.decode(wb_data.text)
        if len(obj) < 1:
            return -1
        return obj[-1]['data']['PvNum']
    
    @staticmethod
    def test(params):
        print('powerful web spider')
        website = '962.net'
        cnTop, alexaTop, classTop, provTop = RglMain.get_all_tops('http://top.chinaz.com/html/site_{0}.html'.format(website))
        print('alexaTop:{0}, cnTop:{1}, classTop:{2}'.format(alexaTop, cnTop, classTop))
        baiduWeight = RglMain.get_baidu_weight(website)
        print('百度权重:{0}'.format(baiduWeight))
        RglMain.get_alexa_ip_num(website)
        RglMain.get_alexa_pv_num(website)
    
    @staticmethod
    def startup(params):
        crack_website_file = 'd:/awork/crack_website.csv'
        recs = []
        with open(crack_website_file, 'r', newline='') as csv_file:
            rows = csv.reader(csv_file, delimiter=',', quotechar='|')
            for row in rows:
                website = str(row[2])
                cnTop, alexaTop, classTop, provTop = RglMain.get_all_tops('http://top.chinaz.com/html/site_{0}.html'.format(website))
                baiduWeight = RglMain.get_baidu_weight(website)
                ip_num = RglMain.get_alexa_ip_num(website)
                pv_num = RglMain.get_alexa_pv_num(website)
                print('website:{0} {1} {2} {3} IP:{4} PV:{5}'.format(website, alexaTop, cnTop, classTop, ip_num, pv_num))
                item = [row[0], row[1], row[2], baiduWeight, alexaTop, cnTop, classTop, ip_num, pv_num]
                recs.append(item)
        
        result_file = 'd:/awork/crack_website_new.csv'
        with open(result_file, 'w', newline='') as csv_file:
            cw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            cw.writerow(['侵权网站', 'URL', '域名', '百度权重', '全球排名', '中国排名', '类目排名', '近一周IP', '近一周PV'])
            for rec in recs:
                print('write:{0}'.format(rec))
                cw.writerow(rec)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
