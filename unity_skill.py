import socket
import numpy as np
from time import sleep
class unity:
    def __init__(self,host="127.0.0.1",port=12333):
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.is_gaming = False
    def connect(self):
        self.client.connect((self.host, self.port))
    
    def close(self):
        self.client.send(bytes("end","utf-8"))
        self.client.close()
    
    def skill(self,skill):
        '''
        light_punch
        punch
        defense
        jump
        damaged
        win
        lose
        '''
        self.client.send(bytes(skill,"utf-8"))

    def is_gaming(self):
         return self.is_gaming

if __name__ == "__main__":
    uni = unity() # 宣告 unity(host,port)
    uni.connect()
    msg = 'punch'
    while(msg!='end'):
        uni.skill(msg)
        print(uni.is_gaming)
        msg = input()
    uni.close() # socket close