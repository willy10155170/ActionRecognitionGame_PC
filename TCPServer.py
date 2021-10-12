import time
from socket import *
import threading
import Lidar_positioning_testing
import ActionPredict
import queue
import numpy as np
import unity_skill

#git

class TCPServer:
    def __init__(self):
        self.client_list = []
        self.player_health = [100, 100]
        self.timer = 90
        self.client_num = 0
        self.game_status = False
        self.unity_server = unity_skill.unity()

    def _handle_client(self, client_socket, client_num):
        if client_num == 0:
            enemy = 1
            enemy_socket = self.client_list[1][0]
        else:
            enemy = 0
            enemy_socket = self.client_list[0][0]
        while True:
            try:
                request = client_socket.recv(1024).decode()
                print(request)
                command = request.split(' ')[0]
                results = request.split(' ')[1].split('\n')
                if command == '0':
                    if '攻擊' in results:
                        self.unity_server.skill('attack')
                        print('keyword detected')
                        text = 'player {player_num} attacking...'.format(player_num=client_num)
                        print(text)
                        self.player_health[enemy] -= 100
                        print(self.player_health)
                        client_msg = '對 player {enemy_num} 發動語音攻擊\n'.format(enemy_num=enemy)
                        client_socket.send(client_msg.encode())
                        enemy_msg = '受到 player {player_num} 的語音攻擊\n'.format(player_num=client_num)
                        enemy_socket.send(enemy_msg.encode())
                    elif '防禦' in results:
                        self.unity_server.skill('defense')
                        print('keyword detected')
                        # print('defending...')
                        text = 'player {player_num} defending...'.format(player_num=client_num)
                        print(text)
                        client_msg = 'player {player_num} 發動語音防禦\n'.format(player_num=client_num)
                        client_socket.send(client_msg.encode())
                        enemy_msg = 'player {enemy_num} 發動語音防禦\n'.format(enemy_num=client_num)
                        enemy_socket.send(enemy_msg.encode())

                elif command == "1":
                    if 'jump' in results:
                        self.unity_server.skill('jump')
                        print('player 跳了起來!')
            except BlockingIOError:
                if self.timer <= 0:
                    self.game_status = True
                    if self.player_health[0] > self.player_health[1]:
                        print('player 0 win\nplayer 1 loss')
                    elif self.player_health[0] == self.player_health[1]:
                        print('draw')
                    else:
                        print('player 1 win\nplayer 0 loss')
                    break
                if self.player_health[0] <= 0:
                    self.game_status = True
                    print('player 0 loss')
                    break
                elif self.player_health[1] <= 0:
                    self.game_status = True
                    print('player 1 loss')
                    break
        print('player{} end'.format(client_num))

    def _handle_send(self, client_socket, client_num):
        while True:
            response = input('input the command:\n') + '\n'
            client_socket.send(response.encode())
        client_socket.close()

    def _handle_new_client(self, client_socket, client_num):
        client_handler = threading.Thread(target=self._handle_client, args=(client_socket, client_num,))
        print(client_num)
        client_handler.start()
        # send_handler = threading.Thread(target=handle_send, args=(client_socket, client_num, ))
        # send_handler.start()
        # send_handler
        print("connect!")

    def start_tcp_server(self):
        serverPort = 6969
        serverScoket = socket(AF_INET, SOCK_STREAM)
        serverScoket.bind(('0.0.0.0', serverPort))
        serverScoket.setblocking(False)
        serverScoket.listen(3)
        print('The server is ready to receive')
        #client_counter = 0
        while True:
            try:
                connectionSocket, addr = serverScoket.accept()
                self.client_list.append([connectionSocket, threading.Thread(target=self._handle_new_client,
                                                                       args=(connectionSocket, self.client_num,)), 100])
                print('player {player} joined'.format(player=self.client_num))
                if self.client_num == 1:
                    print('0')
                    # self.client_list[0][1].start()
                    # self.client_list[1][1].start()
                #client_counter += 1
                self.client_num += 1
            except BlockingIOError:
                if self.client_num == 2:
                    break
                time.sleep(1)
                print('waiting...')
        return 0

    def fight(self):
        print('Game Start')
        predict_model = ActionPredict.Model()
        start = time.time()
        self.timer = 90
        joint_set = []
        self.unity_server.connect()
        # self.client_list[0][1].start()
        # self.client_list[1][1].start()
        frame_data = queue.Queue()
        is_end = queue.Queue()
        Lidar_data = Lidar_positioning_testing.frame_queue(frame_data, is_end)
        lidar_thread = threading.Thread(target=Lidar_data.get_frame_data, args=())
        lidar_thread.start()
        while True:
            if frame_data.empty() is False:
                is_end.put(self.game_status)
                jj = frame_data.get()
                #print(jj)
                joint_set.append(jj)
                if len(joint_set) < 50:
                    print(len(joint_set))
            

            if len(joint_set) == 50:
                JointData = np.array(joint_set)
                JointData = JointData.reshape(1,50,39)
                results = predict_model.predict(JointData)
                print(results)
                self.unity_server.skill('Attack')
                # max_value = max(results)
                # max_index = results.index(max_value)
                # print(max_index)
                joint_set = joint_set[1:]

            end = time.time()
            self.timer = 90 - (end - start)
            print(self.timer)
            if self.timer < 0:
                break
            if self.game_status:
                is_end.put(self.game_status)
                self.unity_server.close()
                break