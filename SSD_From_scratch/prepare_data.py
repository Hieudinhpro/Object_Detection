import os
import urllib.request
import zipfile
import tarfile

''' the module is download data from oxford university and save it to folder name is data'''

data_dir = "./data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")

if not os.path.exists(target_path):
    # Download data from oxford university then save as name target_path
    urllib.request.urlretrieve(url, target_path)

    # read the file tar and extract the data then save as data_dir
    tar = tarfile.TarFile(target_path)
    tar.extractall(data_dir)
    tar.close