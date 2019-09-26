from recognitionface import predict
import socket, threading, time, os
import queue
import sys

SIZE = 1024*500

# delete homonymic file
def checkExsit():
    list = os.listdir('.')
    for iterm in list:
        if iterm == 'image.jpg':
            os.remove(iterm)
            print ('Exsit file has been removed')

def saveImage(sock):
    #print ('Begin to save image ...')
    #checkExsit()
    bo = False
    while True:
        data = sock.recv(SIZE)
        if not data:
            break
        else:
            try:
                if data.decode('utf-8') == 'picture':
                    bo = True
                elif data.decode('utf-8') == 'finish':
                    break
            except:
                with open('./image.jpg', 'ab') as f:
                    f.write(data)
    f.close()
    if bo == True:
        print('Finished saving image ...')
    return bo

def tcplink(sock, addr,q):
	# print connection information
    print ('Accept new connection from %s:%s...' % addr)
	# send hello message
    data = 'hello client'
    sock.send(data.encode('utf-8'))
    #print (sock.recv(SIZE).decode('utf-8'))
    while True:
        b = saveImage(sock)
        if b:
            q.put('-1')
    #recv = None


def output(path):
    boo = 0 #其他
    result = cclass.reco_face([path], 160)
    for i in range(result.__len__()):
        if result[0] == False:
            print("nobody")
            boo = 1 #没有检测到人
            break
        if i == 0:
            continue
        print("monitored" + result[i])
        if result[i]=='HUJUNFENG' or result[i]=='MAWENCHAO' or result[i]=='LUOXIUFENG':
            boo = 2 # 我们三个
            break
    return boo

if __name__ == '__main__':
    q = queue.Queue()
    cclass = predict(r'./models/20170511-185253/20170511-185253.pb',
                              r'./my_classifier.pkl')
    # create socket
    lock = threading.Lock()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# monitor port (you shuold change the IP according to your situation)
    s.bind(('192.168.2.128', 9999))
	# Only one client are permitted
    s.listen(1)
    print ('Waiting for connection...')
    sock, addr = s.accept()
	# build a thread to monitor data
    t = threading.Thread(target=tcplink, args=(sock, addr, q))
    t.start()

    while True:
        if not q.empty():
            q.get()
            tem = False
            si = True
            try:
                tem = output(r'image.JPG')
                checkExsit()
                if tem == 2:
                    sock.send(('open').encode('utf-8'))
                    print('open')
                elif tem == 0:
                    sock.send(('close').encode('utf-8'))
                    print('close')
                elif tem == 1:
                    sock.send(('nobody').encode('utf-8'))
                    print('onbody')
            except:
                sock.send(('failed').encode('utf-8'))
                print('failed')
                checkExsit()

