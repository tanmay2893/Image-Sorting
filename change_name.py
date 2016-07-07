import os
current_directory=os.path.dirname(os.path.abspath(__file__))
x= os.listdir(current_directory)
new_x=[]
for i in x:
    if i.find('.')==-1:
        new_x+=[i]
x=new_x
for i in x:
    path=current_directory+'\\'+i
    j=0
    g=0
    found=0
    changes=[]
    for filename in os.listdir(path):
        t=filename.find('_')
        if t!=-1:
            temp=int(filename[t+1:filename.index('.')])
            if temp>g:
                g=temp
        else:
            found=1
            changes+=[filename]
    if found==0:
        continue
    if g==0:
        for filename in os.listdir(path):
            j+=1
            name=i+'_'+str(j)+'.jpg'
            try:
                os.rename(path+'\\'+filename,path+'\\'+name)
            except:
                continue
    else:
        for filename in changes:
            g+=1
            name=i+'_'+str(g)+'.jpg'
            #raw_input('')
            os.rename(path+'\\'+filename,path+'\\'+name)
                        
