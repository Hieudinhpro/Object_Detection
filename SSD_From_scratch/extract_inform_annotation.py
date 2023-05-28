from lib import *
from make_datapath import make_datapath_list


class Anno_xml(object):
    '''extract information of an XML file, read info such as xmin and xmax ymin and ymax the result are bouding boxes
            of object [[xmin, ymin, xmax, ymax, label_id], ......] , label_id is class off object
            input: path of file, width, height
            output: list of objects has shape [[xmin, ymin, xmax, ymax, label_id],...]'''
    
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        # include image annotation
        ret = []
        # read file xml
        xml = ET.parse(xml_path).getroot()
        
        # get each object from list objects in xlm file using xml.iter('object'):
        for obj in xml.iter('object'):

            # object is difficult to learn to skip using obj.find("difficult") with key "difficult result is text of tag difficult
            difficult = int(obj.find("difficult").text) 
            if difficult == 1:
                continue
            # information for bounding box    
            bndbox = []
            name = obj.find("name").text.lower().strip() # .strip() to cut /n and spaces
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1 # in dataset VOC gia tri bat dau tu 1 1 nen - 1 de ve gia tri 0 0
                if pt == "xmin" or pt == "xmax":
                    pixel /= width # ratio of width
                else:
                    pixel /= height # ratio of height
                # chuyen toa do ve dang ti le  
                bndbox.append(pixel)
            label_id = self.classes.index(name)
            bndbox.append(label_id)
            ret += [bndbox]
        return np.array(ret) #[[xmin, ymin, xmax, ymax, label_id], ......]

if __name__ == "__main__":
    '''test module'''

    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    anno_xml = Anno_xml(classes)

    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)
    idx = 1
    img_file_path = val_img_list[idx]
    img = cv2.imread(img_file_path) # output of cv2.imread are image has [height, width, 3 channels:BGR]
    height, width, channels = img.shape # get size img
    # print("Size img {}, {}, {}".format(height, width, channels))
    # xml_path, width, height
    annotation_infor = anno_xml(val_annotation_list[idx], width, height)
    print(annotation_infor)
