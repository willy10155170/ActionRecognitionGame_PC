import socket
import numpy as np

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
        attack
        defense
        jump
        skill_1
        skill_2
        skill_3
        skill_4
        '''
        self.client.send(bytes(skill,"utf-8"))

    def is_gaming(self):
         return self.is_gaming