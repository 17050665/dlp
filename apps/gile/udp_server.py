from socket import *
from time import ctime

class UdpServer(object):
    host = ''
    port = 8088
    buf_size = 1024
    addr = (host, port)
    
    def startup(host='', port='', buf_size=1024):
        UdpServer.host = host
        UdpServer.port = port
        UdpServer.buf_size = buf_size
        UdpServer.addr = (UdpServer.host, UdpServer.port)
        udp_server = socket(AF_INET, SOCK_DGRAM)
        udpServer.bind(ADDR)
        idx = 1
        while True:
            print('waiting for message...')
            client_data, client_addr = udp_server.recvfrom(buf_size)
            print('received from {0}:{1}'.format(client_addr, client_data))
            udp_server.sendto('message{0}'.format(idx), client_addr)
            idx += 1
        udp_server.close()
        