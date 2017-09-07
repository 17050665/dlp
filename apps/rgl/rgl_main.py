from bs4 import BeautifulSoup
import requests
from apps.rgl.spider_html_render import SpiderHtmlRender

class RglMain(object):
    pc_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'
    pc_cookie = 'UM_distinctid=15dabfd5e91430-0c7e81214924c3-66547728-1fa400-15dabfd5e92894; qHistory=aHR0cDovL3Nlby5jaGluYXouY29tK1NFT+e7vOWQiOafpeivonxodHRwOi8vcmFuay5jaGluYXouY29tK+eZvuW6puadg+mHjeafpeivonxodHRwOi8vdG9vbC5jaGluYXouY29tK+ermemVv+W3peWFt3xodHRwOi8vdG9vbC5jaGluYXouY29tL3Rvb2xzL3VuaWNvZGUuYXNweCtVbmljb2Rl57yW56CB6L2s5o2i; CNZZDATA433095=cnzz_eid%3D1861508059-1504757652-http%253A%252F%252Ftool.chinaz.com%252F%26ntime%3D1504757652; CNZZDATA5082706=cnzz_eid%3D1943077002-1504754189-http%253A%252F%252Ftool.chinaz.com%252F%26ntime%3D1504754189; alexadataitjuzi.com=13%2c576%7c38100%7c48%2c000%7c216%2c000'
    
    post_headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': pc_user_agent,
        'Cookie': pc_cookie
    }
    
    get_headers = {
        'User-Agent': pc_user_agent,
        'Cookie': pc_cookie
    }
    
    @staticmethod
    def startup(params):
        baiduWeight = RglMain.get_baidu_weight()
        print('百度排名：{0}'.format(baiduWeight))
        world_top = RglMain.get_all_tops()
        print('全球排名：{0} 中文排名：'.format(world_top))
        
    @staticmethod
    def get_baidu_weight():
        ''' 获取百度权重 '''
        website = 'www.gamersky.com'
        url = 'http://seo.chinaz.com/{0}'.format(website)
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
    def get_all_tops():
        ''' 获取网站全球排名 '''
        website = 'www.gamersky.com'
        url = 'http://alexa.chinaz.com/{0}'.format(website)
        wb_data = requests.get(url, headers=RglMain.get_headers)
        soup = BeautifulSoup(wb_data.text, 'lxml')
        tag_objs = soup.select('#form > div:nth-of-type(4) > div.row_title.bor-t1s.pt20.clearfix > h4 > em:nth-of-type(2)')
        tag_obj = tag_objs[0]
        # 中国排免 #topRankedSpan > em:nth-child(1)
        cn_ranking_objs = soup.select('#form > div:nth-of-type(4) > div.row_title.bor-t1s.pt20.clearfix > h4 > span')
        print(cn_ranking_objs)
        RglMain.get_tops_ajax('gamersky.com')
        return tag_obj.get_text()
        
    def get_tops_ajax(website):
        url = 'http://search.top.chinaz.com/json/GetTopRanked.asmx/GetTopChinazRand?jsoncallback=?'
        post_data = {'url': website}
        wb_data = requests.post(url, headers=RglMain.post_headers, data=post_data)
        print(wb_data.text)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
