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
        self.right_joint_set = []
        self.left_joint_set = []
        self.client_list = []
        self.client_num = 0
        self.game_status = False
        self.unity_player0 = unity_skill.unity(port=12333)
        self.unity_player1 = unity_skill.unity(port=12334)
        self.lock = threading.Lock()
        self.model = ActionPredict.Model()

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
                    if '挑釁' in results:
                        self._send_action(client_num, 5)
                        print('keyword detected')
                        text = 'player {player_num} 發動挑釁...'.format(player_num=client_num)
                        print(text)
                        client_msg = '對 player {enemy_num} 發動挑釁\n'.format(enemy_num=enemy)
                        client_socket.send(client_msg.encode())
                        enemy_msg = '受到 player {player_num} 的挑釁\n'.format(player_num=client_num)
                        enemy_socket.send(enemy_msg.encode())
                    elif '健美' in results:
                        self._send_action(client_num, 6)
                        print('keyword detected')
                        # print('defending...')
                        text = 'player {player_num} defending...'.format(player_num=client_num)
                        print(text)
                        client_msg = 'player {player_num} 發動健美\n'.format(player_num=client_num)
                        client_socket.send(client_msg.encode())
                        enemy_msg = 'player {player_num} 發動健美\n'.format(player_num=client_num)
                        enemy_socket.send(enemy_msg.encode())

                elif command == "1":
                    if 'jump' in results:
                        self._send_action(client_num, 7)
                        print('player 跳了起來!')
            except BlockingIOError:
                if self.game_status:
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

    def _action_predict(self):
        left_results = None
        right_results = None
        if len(self.left_joint_set) == 20:
            data = np.array(self.left_joint_set)
            try:
                left_results = self.model.predict(data)
                self.left_joint_set = self.left_joint_set[1:]
            except Exception as e:
                print(e)

        if len(self.right_joint_set) == 20:
            data = np.array(self.right_joint_set)
            try:
                right_results = self.model.predict(data)
                self.right_joint_set = self.right_joint_set[1:]
            except Exception as e:
                print(e)

        return left_results, right_results

    def _send_action(self, player, action):
        if player == 0:
            send = self.unity_player0
        else:
            send = self.unity_player1

        if action == 1:
            send.skill("attack")
        elif action == 2:
            send.skill("defense")
        elif action == 3:
            send.skill("skill_1")
        elif action == 4:
            send.skill("skill_2")
        elif action == 5:
            send.skill("skill_3")
        elif action == 6:
            send.skill("skill_4")
        elif action == 7:
            send.skill("jump")

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
        self.unity_player0.connect()
        self.unity_player1.connect()
        self.client_list[0][1].start()
        self.client_list[1][1].start()
        frame_data = queue.Queue()
        is_end = queue.Queue()
        Lidar_data = Lidar_positioning_testing.frame_queue(frame_data, is_end)
        lidar_thread = threading.Thread(target=Lidar_data.get_frame_data, args=())
        lidar_thread.start()
        self.unity_player0.start()
        while True:
            if frame_data.empty() is False:
                is_end.put(self.game_status)
                data = frame_data.get()
                self.left_joint_set.append(data[0])
                self.right_joint_set.append(data[1])
                left_action, right_action = self._action_predict()
                self._send_action(0, left_action)
                self._send_action(1, right_action)

            self.game_status = self.unity_player0.is_gaming()

            if self.game_status:
                is_end.put(self.game_status)
                self.unity_player0.close()
                self.unity_player1.close()
                break
