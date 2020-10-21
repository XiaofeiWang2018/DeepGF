from sklearn.metrics import roc_auc_score
import glob
import os

image_dir = os.listdir('data/train/image/all' )

year_interval_test=[0 for i_ in range(20)]
year_interval_train=[0 for ij in range(20)]
for i in range(len(image_dir)):
    image_sublist = glob.glob('data/train/image/all/' + image_dir[i] + '/' + '*.jpg')
    image_sublist.sort()
    for idx_image in range(5):
        image_now = image_sublist[idx_image]
        image_next = image_sublist[idx_image + 1]
        img_name_now = os.path.split(image_now)[-1]
        img_name_next = os.path.split(image_next)[-1]
        year_now = int(img_name_now[7:11])
        year_next = int(img_name_next[7:11])
        delta_year = year_next - year_now
        if delta_year==0:
            delta_year=1

        year_interval_train[delta_year]+=1

a=1