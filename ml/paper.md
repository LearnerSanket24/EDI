# Title: Lightweight Head Pose Estimation for Real-Time Applications: A Comparative Study with MobileNet and Alternatives

Authors: [Your Name], [Affiliation]

Abstract
We present a lightweight head pose estimation system aimed at real-time human-computer interaction and safety-critical applications. Using folder-based classification with five classes (forward, left, right, up, down), we evaluate multiple backbones including MobileNetV2 (PyTorch and Keras), a custom CNN, and EfficientNetB0 (Keras). Our experiments leverage a curated face-crop dataset and a lightweight CrowdHuman-derived dataset for auxiliary analysis. The MobileNetV2 classifier achieves 65–70% top-1 accuracy on a held-out validation set, with class-dependent performance as revealed by a confusion matrix. We provide an end-to-end pipeline: preprocessing, data augmentation, training, evaluation (precision/recall/F1), and deployment-ready inference. We discuss error sources (class imbalance, label noise, face detection drift, yaw/pitch/roll overlap) and outline improvements via transfer learning, longer training, stronger augmentations, and transformer-based models. Code, training scripts, and artifacts are provided in the project’s ml directory.

1. Introduction
Head pose estimation enables robust interaction and safety monitoring across driver attention monitoring, healthcare, AR/VR, robotics, and surveillance. Real-time constraints on edge devices motivate compact models with low latency and modest memory footprints. We study a lightweight classification-based approach to discretized head pose using resource-efficient backbones, prioritizing deployability while maintaining accuracy.

Contributions
- A practical, reproducible pipeline for five-class head pose estimation with MobileNetV2 and alternatives.
- Detailed training/evaluation scripts for both PyTorch and Keras/TensorFlow ecosystems.
- Empirical analysis with confusion matrix and dataset statistics; discussion on error modes and remedies.
- Guidance for future extensions including transformer-based models and multi-modal inputs.

2. Related Work
Head pose estimation literature spans regression of yaw/pitch/roll and classification over discretized bins. Notable approaches include:
- HopeNet (Ruiz et al., 2018): CNN-based direct angle regression with binned classification auxiliaries.
- FSA-Net (Yang et al., 2019): Feature separation for fine-grained angles.
- 6DRepNet (Hempel et al., 2022): 6D rotation representation for stable pose regression.
- Lightweight backbones (MobileNet, ShuffleNet, EfficientNet) have been adopted for real-time pose or gaze estimation.
- Vision Transformers (ViT/DeiT) and hybrid CNN-Transformer models show strong performance but incur higher compute; recent efficient ViT variants mitigate this gap.

3. Methods
3.1 Problem Formulation
We formulate head pose estimation as 5-way classification: {forward, left, right, down, up}. This discretization simplifies training and enables robust deployment where angular granularity is not required.

3.2 Datasets
- Head-pose face crops: Folder-structured dataset with train/ and val/ splits as required by our scripts. Images are RGB face crops captured from a frontal camera.
- Auxiliary detection dataset: CrowdHuman lightweight subset for multi-person detection analysis (not directly used for head-pose training). Statistics: train images=800, val images=200, total boxes=1,769, mean 1.77 boxes/image (ml/crowdhuman_light_dataset/dataset_statistics.txt).

3.3 Preprocessing and Augmentation
- Resize to 160×160.
- Normalization: ImageNet mean/std for PyTorch; 1/255 scaling for Keras.
- Augmentations: random horizontal flip, color jitter/brightness/contrast, random rotation and zoom, random contrast; modest magnitudes to preserve pose semantics.

3.4 Models
- MobileNetV2 (PyTorch): Pretrained ImageNet backbone; final classifier replaced with a 5-way linear layer (train_head_pose.py).
- MobileNetV2 (Keras): Pretrained, include_top=False, GAP + small MLP head; base frozen initially.
- EfficientNetB0 (Keras): Pretrained backbone with GAP + MLP head (keras_head_pose_model.py).
- Custom CNN (Keras): 5 conv blocks + dense layers.

3.5 Training Setup
- Image size: 160×160; batch size: 32.
- Optimizers: Adam (lr=1e-3) with ReduceLROnPlateau (PyTorch and Keras).
- Schedulers and early stopping (Keras): ReduceLROnPlateau, EarlyStopping; best checkpoint saving.
- Epochs: 10–50 depending on framework; best checkpoint selected by validation accuracy.

3.6 Evaluation Metrics
We compute accuracy, precision, recall, F1-score (macro and per-class), and confusion matrix. For Keras flows, predictions come from model.predict followed by argmax. The same class order is used across systems: [forward, left, right, down, up].

4. Results
4.1 Confusion Matrix
- See ml/confusion_matrix.png (included). The matrix reveals strong bias toward the "left" class and confusion between forward↔left and right↔forward.

4.2 Quantitative Metrics
- Overall accuracy: approximately 65% (val). Class-wise metrics can be computed with the provided evaluation function in train_keras_head_pose.py (classification_report). Precision/recall/F1 reflect class imbalance; forward/left dominate predictions with reduced recall for down/up.

4.3 Training Dynamics
- The Keras training script saves loss/accuracy curves to models/head_pose_<arch>_history.png. These plots typically show rapid convergence in the first 5–10 epochs with validation plateaus thereafter; learning-rate reductions provide minor gains.

4.4 Model Comparison (qualitative)
- MobileNetV2 vs Custom CNN: MobileNetV2 converges faster and achieves higher accuracy at similar input size due to stronger inductive biases and pretrained features.
- MobileNetV2 vs EfficientNetB0: EfficientNetB0 provides marginal gains when unfrozen/fine-tuned but with higher compute; under frozen-backbone regime, performance is similar.

5. Discussion
5.1 Why accuracy saturates near 65–70%
- Dataset size and diversity: limited subjects, lighting, and occlusions reduce generalization.
- Class imbalance: "left" and "forward" dominate; model over-predicts these classes.
- Label granularity: discrete labels fold many similar angles into coarse bins; borderline samples are ambiguous.
- Face detection drift: cascades may crop off-axis faces suboptimally, harming pose cues.

5.2 Improvements
- Data: increase subjects and viewpoints; balance classes; stronger augmentation (CutMix/MixUp, color/blur, coarse dropout); synthetic augmentation via 300W-LP or generating yaw/pitch variations.
- Training: fine-tune pretrained backbones (unfreeze top-N blocks with low lr); longer training with cosine decay; class-weighted loss or focal loss to counter imbalance.
- Models: EfficientNetV2-S, MobileNetV3, or lightweight transformer backbones (DeiT-Ti, MobileViT). For regression, investigate 6DRepNet or HopeNet-style hybrid classification+regression.
- Multi-modal: incorporate depth/IR where available to disambiguate lighting and occlusions; temporal modeling with lightweight RNN/TCN for video.

6. Deployment and Inference
We provide CPU-friendly inference with optional face detection and cropping (ml/inference_head_pose.py). The model expects RGB face crops resized to 160×160 and outputs pose probabilities and top-1 prediction with confidence. For multi-person scenes, first run a person/face detector, then route each face through the head pose classifier.

7. Conclusion and Future Work
We demonstrate a practical lightweight head pose estimation pipeline centered on MobileNetV2 with competitive performance for real-time applications. While current accuracy (~65–70%) suits coarse interaction tasks, safety-critical systems may demand higher reliability. Future work includes: transformer-based lightweight backbones, multi-task learning (pose + landmarks), multi-modal fusion (depth/IR), semi-supervised pretraining, and training on larger curated datasets (BIWI, 300W-LP, AFLW2000, Pandora).

Acknowledgments
We thank open-source contributors of PyTorch, TensorFlow/Keras, and the CrowdHuman dataset.

References
[1] Ruiz, N., Chong, E., & Rehg, J. M. (2018). Fine-Grained Head Pose Estimation Without Keypoints (HopeNet).
[2] Yang, T.-Y., Huang, Y.-H., Lin, Y.-Y., et al. (2019). FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation.
[3] Hempel, T., Abdelrahman, A. A., et al. (2022). 6D Rotation Representation for Unconstrained Head Pose Estimation (6DRepNet).
[4] Sandler, M., Howard, A., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
[5] Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
[6] Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.

