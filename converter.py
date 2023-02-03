import xml.etree.ElementTree as ET
import json
import os

def parse_xml(xml_file):
    # parse the XML file into a dictionary
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_width = int(root.find("size").find("width").text)
    image_height = int(root.find("size").find("height").text)

    shapes = []
    for obj in root.iter("object"):
        label_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox_x = int(bndbox.find("xmin").text)
        bbox_y = int(bndbox.find("ymin").text)
        bbox_width = int(bndbox.find("xmax").text) - bbox_x
        bbox_height = int(bndbox.find("ymax").text) - bbox_y

        shapes.append({
            "label": label_name,
            "points": [[bbox_x, bbox_y], [bbox_x + bbox_width, bbox_y + bbox_height]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        })

    return {
        "version": "5.1.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
    }

# convert the XML files to LabelMe JSON format
xml_dir = "C:/Users/darks/Desktop/Emotion AI/Emotion Recognition/FER2013/Test_Face_Detector/Test_Face_Detector/workspace/images/test"
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        labelme_json = parse_xml(os.path.join(xml_dir, xml_file))

        with open("{}.json".format(os.path.splitext(xml_file)[0]), "w") as jsonfile:
            json.dump(labelme_json, jsonfile, indent=2)