import os
import xml.etree.ElementTree as ET
import glob


def count_num(indir):
    label_list = []
    # 提取xml文件列表
    # 不要改变当前目录，直接使用glob来查找路径下的文件
    annotations = glob.glob(os.path.join(indir, '*.xml'))  # 获取xml文件列表

    label_dict = {}  # 新建字典，用于存放各类标签名及其对应的数目
    for i, file in enumerate(annotations):  # 遍历xml文件
        # actual parsing
        with open(file, encoding='utf-8') as in_file:
            tree = ET.parse(in_file)
            root = tree.getroot()

            # 遍历文件的所有标签
            for obj in root.iter('object'):
                name = obj.find('name').text
                if name in label_dict:
                    label_dict[name] += 1  # 如果标签不是第一次出现，则+1
                else:
                    label_dict[name] = 1  # 如果标签是第一次出现，则将该标签名对应的value初始化为1

    # 打印结果
    print("各类标签的数量分别为：")
    for key, value in label_dict.items():
        print(f"{key}: {value}")
        label_list.append(key)
    
    print("标签类别如下：")
    print(label_list)


if __name__ == '__main__':
    # xml文件所在的目录，修改此处
    indir = 'data/Annotations'  # 确保路径正确
    count_num(indir)  # 调用函数统计各类标签数目
