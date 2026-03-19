from pycocotools import mask
from skimage import measure
import json
import shutil
import itertools
import numpy as np
from simplification.cutil import simplify_coords_vwp
import os, cv2, copy
from distinctipy import distinctipy
import re

def mask_touch_border(mask: np.ndarray) -> bool:
    return (
        mask[0, :].any() or
        mask[-1, :].any() or
        mask[:, 0].any() or
        mask[:, -1].any()
    )


def has_holes(mask: np.ndarray) -> bool:
    """
    检测 mask 是否包含内部孔洞（如圆环）
    使用 RETR_CCOMP 可检测父子轮廓关系
    """
    mask_uint8 = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask_uint8,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return False

    # hierarchy[0][i][3] != -1 表示该轮廓有父轮廓 → 是洞
    for h in hierarchy[0]:
        if h[3] != -1:
            return True
    return False
    

def safe_find_contours(mask: np.ndarray):
    """
    对 mask 做 padding，避免贴边轮廓断裂
    """
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded, 0.5)
    return [c - 1 for c in contours]  # 映射回原坐标


def init_coco(dataset_folder, image_names, categories, coco_json_path):
    coco_json = {
        "info": {
            "description": "SAM Dataset",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "Sam",
            "date_created": "2021/07/01",
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for i, category in enumerate(categories):
        coco_json["categories"].append(
            {"id": i, "name": category, "supercategory": category}
        )
    for i, image_name in enumerate(image_names):
        im = cv2.imread(os.path.join(dataset_folder, image_name))
        coco_json["images"].append(
            {
                "id": i,
                "file_name": image_name,
                "width": im.shape[1],
                "height": im.shape[0],
            }
        )
    with open(coco_json_path, "w") as f:
        json.dump(coco_json, f)


def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans


def unbunch_coords(coords):
    return list(itertools.chain(*coords))


def bounding_box_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h


def parse_mask_to_coco(
    image_id,
    anno_id,
    image_mask,
    category_id,
    poly=True,
):
    image_mask = image_mask.astype(np.uint8)

    # bbox
    x, y, w, h = bounding_box_from_mask(image_mask)

    # ✅ 使用真实 mask area（不是 bbox）
    area = float(np.sum(image_mask))

    annotation = {
        "id": anno_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(w), float(h)],
        "area": area,
        "iscrowd": 0,
        "segmentation": [],
    }

    # --------------------------------
    # ⭐ 1️⃣ 贴边 或 有孔洞 → 强制 RLE
    # --------------------------------
    if mask_touch_border(image_mask) or has_holes(image_mask):
        poly = False

    # --------------------------------
    # 2️⃣ RLE（最稳）
    # --------------------------------
    if not poly:
        rle = mask.encode(np.asfortranarray(image_mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        annotation["segmentation"] = rle
        annotation["iscrowd"] = 1   # COCO 规范
        return annotation

    # --------------------------------
    # 3️⃣ Polygon（安全版）
    # --------------------------------
    contours = safe_find_contours(image_mask)

    for contour in contours:
        if contour.shape[0] < 3:
            continue

        contour = np.flip(contour, axis=1)
        seg = contour.ravel().tolist()

        sc = bunch_coords(seg)
        sc = simplify_coords_vwp(sc, 2)
        sc = unbunch_coords(sc)

        if len(sc) < 6:
            continue

        if sc[:2] != sc[-2:]:
            sc += sc[:2]

        annotation["segmentation"].append(sc)

    # --------------------------------
    # 4️⃣ polygon 失败 → fallback RLE
    # --------------------------------
    if len(annotation["segmentation"]) == 0:
        rle = mask.encode(np.asfortranarray(image_mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        annotation["segmentation"] = rle
        annotation["iscrowd"] = 1

    return annotation



class DatasetExplorer:
    def __init__(self, dataset_folder, categories=None, coco_json_path=None):
        self.dataset_folder = dataset_folder

        def natural_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]

        self.image_names = sorted(
            [f for f in os.listdir(os.path.join(dataset_folder, "images"))
             if f.endswith((".jpg", ".png"))],
            key=natural_key
        )


        self.coco_json_path = coco_json_path

        if not os.path.exists(coco_json_path):
            self.__init_coco_json(categories)

        with open(coco_json_path, "r") as f:
            self.coco_json = json.load(f)
        
        self.coco_json["images"] = sorted(self.coco_json["images"],key=lambda x: natural_key(x["file_name"]))

        # 初始化 annotations_by_image_id
        self.annotations_by_image_id = {}
        max_id = 0
        for ann in self.coco_json["annotations"]:
            self.annotations_by_image_id.setdefault(ann["image_id"], []).append(ann)
            if ann["id"] > max_id:
                max_id = ann["id"]
        self.global_annotation_id = max_id + 1

        self.categories = [c["name"] for c in self.coco_json["categories"]]
        colors = distinctipy.get_colors(len(self.categories))
        self.category_colors = [tuple(int(255 * c) for c in col) for col in colors]

    def __init_coco_json(self, categories):
        appended_image_names = [
            os.path.join("images", name) for name in self.image_names
        ]
        init_coco(
            self.dataset_folder, appended_image_names, categories, self.coco_json_path
        )

    def get_colors(self, category_id):
        return self.category_colors[category_id]
    
    def get_categories(self):
        return self.categories

    def get_num_images(self):
        return len(self.image_names)

    def get_image_data(self, image_id):
        image_name = self.coco_json["images"][image_id]["file_name"]
        image_path = os.path.join(self.dataset_folder, image_name)
        embedding_path = os.path.join(
            self.dataset_folder,
            "embeddings",
            os.path.splitext(os.path.split(image_name)[1])[0] + ".npy",
        )
        image = cv2.imread(image_path)
        image_bgr = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_embedding = np.load(embedding_path)
        return image, image_bgr, image_embedding

    def __add_to_our_annotation_dict(self, annotation):
        image_id = annotation["image_id"]
        if image_id not in self.annotations_by_image_id:
            self.annotations_by_image_id[image_id] = []
        self.annotations_by_image_id[image_id].append(annotation)
    
    def __delet_to_our_annotation_dict(self, image_id):
        self.annotations_by_image_id[image_id].pop(-1)

    def get_annotations(self, image_id, return_colors=False):
        if image_id not in self.annotations_by_image_id:
            return [], []
        cats = [a["category_id"] for a in self.annotations_by_image_id[image_id]]
        colors = [self.category_colors[c] for c in cats]
        if return_colors:
            return self.annotations_by_image_id[image_id], colors
        return self.annotations_by_image_id[image_id]

    def add_annotation(self, image_id, category_id, mask, poly=True):
        if mask is None:
            return
        ann = parse_mask_to_coco(image_id, self.global_annotation_id, mask, category_id, poly)
        self.global_annotation_id += 1

        self.annotations_by_image_id.setdefault(image_id, []).append(ann)
        self.coco_json["annotations"].append(ann)

    def delet_annotation(self, image_id, annotation_id):
        """
        删除指定 image_id 下的指定 annotation_id
        """
        if image_id not in self.annotations_by_image_id:
            return
        # 删除字典里的对象
        self.annotations_by_image_id[image_id] = [a for a in self.annotations_by_image_id[image_id] if
                                                  a["id"] != annotation_id]
        if not self.annotations_by_image_id[image_id]:
            del self.annotations_by_image_id[image_id]
        # 删除 coco_json
        self.coco_json["annotations"] = [a for a in self.coco_json["annotations"] if a["id"] != annotation_id]

    def clear_annotations_for_image(self, image_id):
        """
        删除指定 image_id 下的所有标注
        """
        if image_id in self.annotations_by_image_id:
            del self.annotations_by_image_id[image_id]
        self.coco_json["annotations"] = [a for a in self.coco_json["annotations"] if a["image_id"] != image_id]

    def save_annotation(self):
        # 保存前可选清理非法 annotation
        valid_annotations = []
        for ann in self.coco_json["annotations"]:
            seg = ann["segmentation"]
            if isinstance(seg, list):
                # polygon 至少 3 点
                if all(len(s) >= 6 for s in seg):
                    valid_annotations.append(ann)
            else:
                valid_annotations.append(ann)
        self.coco_json["annotations"] = valid_annotations

        with open(self.coco_json_path, "w") as f:
            json.dump(self.coco_json, f, indent=2)
