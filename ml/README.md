# Head Pose Model (MobileNetV2)

## Dataset Format
Folder-based classification with five classes:
```
data/
  train/
    forward/ *.jpg
    left/    *.jpg
    right/   *.jpg
    down/    *.jpg
    up/      *.jpg
  val/
    forward/ *.jpg
    left/    *.jpg
    right/   *.jpg
    down/    *.jpg
    up/      *.jpg
```

Images should be face crops (frontal camera). Aim for class balance; 500–2,000 images per class is a good start.

## Train
```
cd Project/ml
python train_head_pose.py --data_dir path/to/data --out ../backend/models/head_pose_mnv2.pt --epochs 12 --batch_size 32 --img_size 160
```

This produces `Project/backend/models/head_pose_mnv2.pt`.

## Integrate
The backend will auto-load the model if the file exists. Start the backend after training or place the file in `Project/backend/models/head_pose_mnv2.pt`.


