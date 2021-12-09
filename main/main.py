import TCPServer


def main():
    print('press any key to start the game:')
    #input()
    test = TCPServer.TCPServer()
    test.start_tcp_server()
    test.fight()


if __name__ == "__main__":
    main()
