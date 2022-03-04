import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to the labeling server…")
socket = context.socket(zmq.REP)
socket.connect("tcp://127.0.0.1:5555")

while True:
    message = socket.recv_string()
    print(message)
    if message == 'Finish':
        break
    else:
        socket.send_string(input())
