#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <sys/socket.h>  
#include <netinet/in.h>  
#include <arpa/inet.h>  
#include <netdb.h>  
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/fb.h>

int get_fb_fix_screeninfo(int fd, struct fb_fix_screeninfo** finfo);
int get_fb_var_screeninfo(int fd, struct fb_var_screeninfo** vinfo);

int read_frame_buffer()
{
    int fd;
    int rst;
    struct fb_fix_screeninfo* finfo = NULL;
    struct fb_var_screeninfo* vinfo = NULL;
    fd = open("/dev/graphics/fb0", O_RDONLY);
    if (fd < 0) 
    {
        printf("Fail to open frame buffer!\r\n");
        return -1;
    }
    rst = get_fb_fix_screeninfo(fd, &finfo);
    rst = get_fb_var_screeninfo(fd, &vinfo);
    printf("v0.5 type:%d; line_length:%d\r\n", finfo->type, finfo->line_length);
    printf("red: offset:%d len:%d type:%d\r\n", vinfo->red.offset, vinfo->red.length, vinfo->red.msb_right);
    printf("Gree: offset:%d, len:%d, type:%d\r\n", vinfo->green.offset, vinfo->green.length, vinfo->green.msb_right);
    printf("Blue: offset:%d, len:%d, type:%d\r\n", vinfo->blue.offset, vinfo->blue.length, vinfo->blue.msb_right);
    printf("A: offset:%d, len:%d, type:%d\r\n", vinfo->transp.offset, vinfo->transp.length, vinfo->transp.msb_right);
    printf("screen: width:%d, height:%d\r\n", vinfo->xres, vinfo->yres);
    printf("virtual screen: width:%d, height:%d\r\n", vinfo->xres_virtual, vinfo->yres_virtual);
    printf("virtual offset: xoffset:%d, yoffset:%d\r\n", vinfo->xoffset, vinfo->yoffset);
    free(finfo);
    free(vinfo);
    close(fd);
    return 0;
}

int get_fb_fix_screeninfo(int fd, struct fb_fix_screeninfo** finfo)
{
    int ret;
    *finfo = (struct fb_fix_screeninfo*)malloc(sizeof(struct fb_fix_screeninfo));
    fd = open("/dev/graphics/fb0", O_RDONLY);
    ret = ioctl(fd, FBIOGET_FSCREENINFO, *finfo);
    if (ret < 0)
    {
        printf("Fail to get screen info!\r\n");
        return -2;
    }
    return 0;
}

int get_fb_var_screeninfo(int fd, struct fb_var_screeninfo** vinfo)
{
    int ret;
    *vinfo = (struct fb_var_screeninfo*)malloc(sizeof(struct fb_var_screeninfo));
    //static struct fb_var_screeninfo vinfo;
    // 打开Framebuffer设备  
    fd = open("/dev/graphics/fb0", O_RDONLY);
    // 获取FrameBuffer 的 variable info 可变信息  
    ret = ioctl(fd, FBIOGET_VSCREENINFO, *vinfo);  
    if(ret < 0 )  
    {  
        printf("======Cannot get variable screen information.");  
        close(fd);  
        return -2;  
    }  
    return 0;
}

int get_frame_buffer_rgb(int fd, byte* p_rgb)
{
    printf("get frame buffer image data");
    return 0;
}