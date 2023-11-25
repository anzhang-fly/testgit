import os
import shutil

path1 = 'VOCdevkit/VOC2007/Annotations/'
path2 = 'VOCdevkit/VOC2007/JPEGImages/'

txtout = '1/'
imgout = '2/'

file = os.listdir(path1)
for filename in file:
    path3 = os.path.join(path1, filename)
    path5 = os.path.join(txtout, filename)

    if len(filename) > 11:
        shutil.move(path3, path5)

file1 = os.listdir(path2)
for filename in file1:
    path4 = os.path.join(path2, filename)
    path6 = os.path.join(imgout, filename)
    if len(filename) > 11:
        shutil.move(path4, path6)
