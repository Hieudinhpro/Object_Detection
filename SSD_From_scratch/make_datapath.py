from lib import *

def make_datapath_list(root_path):
    '''open file in directory and read id of image then return list of images'''

    image_path_template = osp.join(root_path, "JPEGImages", "%s.jpg")
    annotation_path_template = osp.join(root_path, "Annotations", "%s.xml")

    train_id_names = osp.join(root_path, "ImageSets/Main/train.txt")
    val_id_names = osp.join(root_path, "ImageSets/Main/val.txt")

    train_img_list = list()
    train_annotation_list = list()

    val_img_list = list()
    val_annotation_list = list()

    # open file and readlines the save id image into train_img_list and val_img_list
    for line in open(train_id_names):
        file_id = line.strip() # xoá ký tự xuống dòng, delete space
        img_path = (image_path_template % file_id) # truyen tham so file_id into image_path 
        anno_path = (annotation_path_template % file_id)

        train_img_list.append(img_path)
        train_annotation_list.append(anno_path)
    
    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_template % file_id)

        val_img_list.append(img_path)
        val_annotation_list.append(anno_path)

    return train_img_list, train_annotation_list, val_img_list, val_annotation_list


if __name__ == "__main__":
    ''' test module bang cach dat ham main va goi ra de test '''

    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    print(len(train_img_list))
    print(train_img_list[0])