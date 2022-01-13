import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from AirplaneDataset import AirplaneDetDataset
from Averager import Averager


def ImageShow(img, bboxes):
    for det in bboxes:
        bbox = np.array(det[:4]).astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)
        label_text = str("airplane")
        cv2.putText(img, label_text, (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    im2 = img[:, :, ::-1]
    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    plt.imshow(im2)
    plt.show()


def get_object_detection_model(num_classes):
    # 载入一个在COCO数据集上预训练好的faster rcnn 模型，backbone为resnet50，neck网络使用fpn
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取最后分类的head的输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 将最后分类的head从原始的COCO输出81类替换为我们现在输入的num_classes类，注意这里的num_classes=实际的类别数量+1，1代表背景
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == '__main__':
    with open("datasets/mini_airplane/annotations/mini_airplane_train.json", "r") as f:
        labels = json.load(f)

    # print("images:")
    # print(labels["images"][0])
    # print("annotations:")
    # print(labels["annotations"][0])
    # print("categories:")
    # print(labels["categories"])

    imagename2id = {}
    imageid2bbox = {}
    imageid2anno = {}
    imagename2imageinfo = {}
    for image in labels["images"]:
        imagename2id[image["file_name"]] = image["id"]
        imagename2imageinfo[image["file_name"]] = image
    for anno in labels["annotations"]:
        if anno["image_id"] not in imageid2bbox:
            imageid2bbox[anno["image_id"]] = []
        imageid2bbox[anno["image_id"]].append(anno["bbox"])
        if anno["image_id"] not in imageid2anno:
            imageid2anno[anno["image_id"]] = []
        imageid2anno[anno["image_id"]].append(anno)

    img = cv2.imread('datasets/mini_airplane/images/COCO_val2014_000000253223.jpg')
    bboxes = imageid2bbox[imagename2id['COCO_val2014_000000253223.jpg']]
    ImageShow(img, bboxes)

    # 划分训练集和验证集
    np.random.seed(123)
    train_image_names = np.random.choice(list(imagename2id.keys()), int(len(imagename2id.keys()) * 0.8), False)
    val_image_names = list(set(imagename2id.keys()) - set(train_image_names))
    print("train_image_names lenght: ", len(train_image_names))  # 79
    print("val_image_names lenght: ", len(val_image_names))  # 20

    # 划分原始的label
    train_images = []
    train_annotations = []
    for imgname in train_image_names:
        train_images.append(imagename2imageinfo[imgname])
        image_id = imagename2id[imgname]
        train_annotations.extend(imageid2anno[image_id])
    train_instance = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": labels["categories"]
    }

    dirs = r'temp/mini_airplane/annotations'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    with open("temp/mini_airplane/annotations/train.json", "w") as f:
        json.dump(train_instance, f, indent=2)
    val_images = []
    val_annotations = []
    for imgname in val_image_names:
        val_images.append(imagename2imageinfo[imgname])
        image_id = imagename2id[imgname]
        val_annotations.extend(imageid2anno[image_id])
    val_instance = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": labels["categories"]
    }
    with open("temp/mini_airplane/annotations/val.json", "w") as f:
        json.dump(val_instance, f, indent=2)

    if not os.path.exists('./temp/mini_airplane/images'):
        shutil.copytree('./datasets/mini_airplane/images', './temp/mini_airplane/images')
    dataset = AirplaneDetDataset('./temp/mini_airplane', 'train.json')
    print(dataset[0])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 构建我们的基于faster rcnn的飞机检测模型，类别数量如上所讲述=1+1=2
    model = get_object_detection_model(2).to(device)


    # print(model)

    # collate_fn函数用于将一个batch的数据转换为tuple格式
    def collate_fn(batch):
        return tuple(zip(*batch))


    # 使用上面的AirplaneDetDataset来定义训练和验证的数据类
    train_dataset = AirplaneDetDataset('temp/mini_airplane', 'train.json')
    valid_dataset = AirplaneDetDataset('temp/mini_airplane', 'val.json')
    # 定义训练data_loader
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    # 定义验证data_loader
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    params = [p for p in model.parameters() if p.requires_grad]
    # 建立sgd 优化器
    optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = None
    # 一共训练8个epoch
    num_epochs = 8  # 8

    # resnet50模型过大，内存不足
    loss_hist = Averager()
    itr = 1

    # 按照每一个epoch循环训练
    for epoch in range(num_epochs):
        loss_hist.reset()
        # 从data loader 中采样每个batch数据用于训练
        for images, targets, _ in train_data_loader:

            # 将image 和 label 挪到GPU中
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 数据进入模型并反向传播得到loss
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            # 更新模型参数
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 10 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1
            if itr > 10: break;

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch} loss: {loss_hist.value}")

    model.eval()

    itr = 1
    for images, _, imgname in iter(valid_data_loader):
        images = list(img.to(device) for img in images)
        sample = images[0].permute(1, 2, 0).cpu().numpy()
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        #     fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        boxes = outputs[0]['boxes'].data.cpu().numpy()
        scores = outputs[0]['scores'].data.cpu().numpy()

        boxes = boxes[scores >= 0.5].astype(np.int32)
        for box in boxes:
            cv2.rectangle(np.ascontiguousarray(sample),
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (220, 0, 0), 2)

        plt.imshow(sample)
        plt.savefig('./temp/' + str(imgname[0]))
        plt.show()
        itr += 1
        if itr > 5:
            break;