from fastai.vision.all import *
import time
import cv2

def label_func(x): return x.parent.name

def run():
    path = Path("E:/nsfw-sfwFinal/")
    fnames = get_image_files(path)
    print(f"Total Images:{len(fnames)}")


    dls = ImageDataLoaders.from_path_func(path, fnames, label_func,bs=16, num_workers=0)
    learn = cnn_learner(dls, resnet18, metrics=error_rate)
    print("Loaded")
    learn.fine_tune(5, base_lr=1.0e-03)

    learn.export()


if __name__ == '__main__':
    run()