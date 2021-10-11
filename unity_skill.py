import socket
import numpy as np
class unity:
    def __init__(self,host="127.0.0.1",port=12333):
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.client.connect((self.host, self.port))
    
    def close(self):
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

    def _test(self):
        print(self.client)
        self.client.send(bytes("Hello World!","utf-8"))
        
if __name__ == "__main__":
    uni = unity() # 宣告 unity(host,port)
    uni.connect() # socket connect
    uni.skill("light_punch")
    uni.skill("damaged")
    uni.skill("lose")
    uni.close() # socket close