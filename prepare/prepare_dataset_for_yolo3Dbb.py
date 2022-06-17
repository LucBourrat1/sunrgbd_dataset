import os
from glob import glob
import json
import shutil

def from_xy_to_yolo(rectangle):
    x_min = int(round(min(rectangle[0]),0))
    x_max = int(round(max(rectangle[0]),0))
    y_min = int(round(min(rectangle[1]),0))
    y_max = int(round(max(rectangle[1]),0))

    c_x = (x_max + x_min)//2
    c_y = (y_max + y_min)//2
    width = (x_max - x_min)
    height = (y_max - y_min)
    return [c_x, c_y, width, height]


def main():

    IMAGES_FOLDER = "../dataset/images"
    LABELS_FOLDER = "../dataset/labels"
    SET_OBJECTS_FOR_MAPPING = set()

    path = "../sunrgbd/OFFICIAL_SUNRGBD/SUNRGBD"
    img_paths = glob(path+"/*/*/*/image/*") + glob(path+"/*/*/*/*/*/image/*")
    json_paths = ["/".join(img_path.split("/")[:-2]) + "/annotation2D3D/index.json" for img_path in img_paths]    

    idx = 0
    nb_with_door = 0
    nb_with_window = 0
    for img_path, json_path in zip(img_paths, json_paths):
        door_in_image = 0
        window_in_image = 0

        print(f"image n°{idx}")

        # extract 2d bboxes and corresponding object names
        if not os.path.exists(json_path):
            continue
        with open(json_path,"r") as f:
            annotations = json.load(f)
        if len(annotations['frames'][0]) == 0:
            continue
        polygons = [[element['x'],element['y']] for element in annotations['frames'][0]['polygon']]
        yolo_boxes = [from_xy_to_yolo(rec) for rec in polygons]
        object_ids = [element['name'] for element in annotations['objects'] if element != None]

        # add all objects from this image to the global set for mapping
        for object in object_ids:
            SET_OBJECTS_FOR_MAPPING.add(object)
            if "door" in object:
                nb_with_door += 1
                door_in_image = 1
            if "window" in object:
                nb_with_window += 1
                window_in_image = 1

        # if it contains a door, save the image and corresponding labels into the output dataset
        if window_in_image == 1 or door_in_image == 1:
        #if 1 == 1:
            img_filename = os.path.join(IMAGES_FOLDER, f"img_{idx}.jpg")
            labels_filename = os.path.join(LABELS_FOLDER, f"img_{idx}.txt")
            shutil.copy(img_path, img_filename)
            with open (labels_filename, "w") as f:
                for object, box in zip(object_ids, yolo_boxes):
                    if "door" in object or "window" in object:
                    #if 1 == 1:
                        f.write(f"{object} {box[0]} {box[1]} {box[2]} {box[3]}\n")

            idx += 1

    # construct the mapping dictionary and save it in dataset folder
    mapping = {}
    id_object = 0
    for object in SET_OBJECTS_FOR_MAPPING:
        mapping[id_object] = object
        id_object += 1

    with open("../dataset/mapping.json", "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"nb of images with a door: {nb_with_door}")
    print(f"nb of images with a window: {nb_with_window}")

    return


if __name__ == "__main__":
    main()