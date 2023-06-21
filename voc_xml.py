"""Get gt_labels and labels_difficult from xml file."""
import xml.etree.ElementTree as ET

root = ET.fromstring("/root/filescripts/000001.xml")

labels, labels_difficult = set(), set()
for obj in root.findall('object'):
    label_name = obj.find('name').text
    # in case customized dataset has wrong labels
    # or CLASSES has been override.
    # if label_name not in self.CLASSES:
    #     continue
    # label = self.class_to_idx[label_name]
    # difficult = int(obj.find('difficult').text)
    # if difficult:
    #     labels_difficult.add(label)
    # else:
    #     labels.add(label)