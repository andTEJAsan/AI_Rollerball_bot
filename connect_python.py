import socket
import sys
import os

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 8183)); #the localhost address.
print(sys.argv[1], sys.argv[2]);
ip = sys.argv[1]; 
port = int(sys.argv[2]);
sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
cnt = 0;
while(cnt < 1):
    try:
        sock2.connect((ip, port)); 
        cnt+=1;
    except:
        cnt-=1; 
        
print("connected to server at ", ip); 
sock.listen(2); 
client, addr = sock.accept();
while True:
    val = sock2.recv(1024); 
    print(val.decode()); 
    sock.send(val); 