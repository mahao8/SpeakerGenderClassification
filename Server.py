#coding=utf-8
from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi
from Predict import *
import os
preFileName = ""
count = 1
class   PostHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global preFileName
        global count
        count+=1
        if os.path.exists(preFileName):
            os.remove(preFileName)   #删除文件
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     }
        )
        self.send_response(200)
        self.end_headers()
        self.wfile.write(('Client: %s\n' % str(self.client_address)).encode('utf-8'))
        self.wfile.write(('User-agent: %s\n' % str(self.headers['user-agent'])).encode('utf-8'))
        self.wfile.write(('Path: %s\n'%self.path).encode('utf-8'))
        self.wfile.write('Form data:\n'.encode('utf-8'))
        for field in form.keys():
            field_item = form[field]
            filename = "predict_dir/Random/MAHH0/"+field_item.filename

            filevalue  = field_item.value
            filesize = len(filevalue)#文件大小(字节)
            print(len(filevalue))
            print(filename)
            preFileName = filename
            #with open(filename.decode('utf-8'),'wb') as f:
            with open(filename,'wb') as f:
                f.write(filevalue)
                ret = predictGender("predict_dir/Random")
                #ret = predictGenderAge("predict_dir/Random")
            fileCach = "Cache/"+str(count)+"_"+ret+"_"+field_item.filename
            print(ret)
            with open(fileCach,'wb') as f:
                f.write(filevalue)
        self.wfile.write(('Class:{}'.format(ret)).encode('utf-8'))
        return

def StartServer():
    from http.server import HTTPServer
    sever = HTTPServer(("",8080),PostHandler)
    sever.serve_forever()

if __name__=='__main__':
    StartServer()
