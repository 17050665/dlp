import time
import numpy as np
import cv2
from PIL import ImageGrab
import pyHook
from apps.esp.hdw.c_key_board import CKeyBoard as CKeyBoard
from apps.esp.hdw.c_mouse import CMouse as CMouse

class EspMain(object):
    @staticmethod
    def startup(params):
        print('电子竞技平台')
        EspMain.mouse = CMouse()
        EspMain.screen_record()
        
    @staticmethod
    def emulate_kb():
        CKeyBoard.press_key(CKeyBoard.DIK_H)
        CKeyBoard.press_key(CKeyBoard.DIK_E)
        CKeyBoard.press_key(CKeyBoard.DIK_L)
        CKeyBoard.press_key(CKeyBoard.DIK_L)
        CKeyBoard.press_key(CKeyBoard.DIK_O)
        CKeyBoard.press_key(CKeyBoard.DIK_SPACE)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_O)
        CKeyBoard.press_key(CKeyBoard.DIK_R)
        CKeyBoard.press_key(CKeyBoard.DIK_L)
        CKeyBoard.press_key(CKeyBoard.DIK_D)
        
    @staticmethod
    def emulate_mouse():
        EspMain.mouse.click((500, 500), 'right')
    
    @staticmethod
    def screen_record(): 
        last_time = time.time()
        epoch = 0
        while(True):
            # 800x600 windowed mode
            epoch += 1
            if 50 == epoch:
                EspMain.emulate_kb()
            if 80 == epoch:
                EspMain.emulate_mouse()
            printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            last_time = time.time()
            new_screen = printscreen # EspMain.process_img(printscreen)
            cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
            mouse_pos = EspMain.mouse.get_position()
            mouse_lbv = EspMain.mouse._get_button_value('left')
            mouse_mbv = EspMain.mouse._get_button_value('middle')
            mouse_rbv = EspMain.mouse._get_button_value('right')
            mouse_bv = (mouse_lbv, mouse_mbv, mouse_rbv)
            kb_val = cv2.waitKey(25) & 0xFF
            print('loop took {0} seconds; {1}; {2}; {3}'.format(time.time()-last_time, mouse_pos, mouse_bv, kb_val))
            if kb_val == ord('q'):
                cv2.destroyAllWindows()
                break
    
    @staticmethod
    def process_img(image):
        original_image = image
        # convert to gray
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # edge detection
        processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
        return processed_img