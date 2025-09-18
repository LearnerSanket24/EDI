# Research Paper Compilation Guide

## Files Generated
- `paper.tex` - Main LaTeX manuscript
- `refs.bib` - BibTeX references file
- `paper.md` - Markdown version (human-readable)
- `confusion_matrix.png` - Confusion matrix figure (already exists)

## Compiling the LaTeX Paper

### Requirements
- LaTeX distribution (TeX Live, MikTeX, or similar)
- BibTeX for references

### Compilation Commands
```bash
# Navigate to ml directory
cd ml

# Compile LaTeX with bibliography
pdflatex paper.tex
bibtex paper
pdflatex paper.tex  
pdflatex paper.tex

# Or use latexmk for automated compilation
latexmk -pdf paper.tex
```

### Alternative: Online LaTeX Editors
- Upload `paper.tex`, `refs.bib`, and `confusion_matrix.png` to:
  - Overleaf (https://www.overleaf.com/)
  - ShareLaTeX
  - Other online LaTeX editors

## Key Technical Details Included

### 1. Model Architectures
- **MobileNetV2 (PyTorch)**: Pretrained backbone + 5-way linear classifier
- **MobileNetV2 (Keras)**: Pretrained, frozen base + GAP + MLP head  
- **EfficientNetB0 (Keras)**: Similar architecture to MobileNetV2
- **Custom CNN (Keras)**: 5 conv blocks + dense layers with batch norm

### 2. Dataset Statistics
From `crowdhuman_light_dataset/dataset_statistics.txt`:
- Train: 800 images, 1417 boxes (1.77 boxes/image)
- Val: 200 images, 352 boxes (1.76 boxes/image)  
- Total: 1000 images, 1769 boxes

### 3. Training Configuration
- Image size: 160×160
- Batch size: 32
- Optimizer: Adam (lr=1e-3)
- Scheduler: ReduceLROnPlateau
- Epochs: 10-50 depending on framework
- Best checkpoint selection by validation accuracy

### 4. Augmentation Strategy
- Random horizontal flip (p=0.5)
- Color jitter (brightness/contrast/saturation/hue)
- Random rotation (0.1)
- Random zoom (0.1)
- Normalization: ImageNet mean/std (PyTorch) or /255 (Keras)

## Results Summary

### Performance Metrics
- **Overall Accuracy**: ~65-70% (validation)
- **Strong classes**: Forward, Left (higher precision)
- **Weak classes**: Up, Down (lower recall due to class imbalance)

### Confusion Matrix Analysis
From `confusion_matrix.png`:
- Strong diagonal for Left class (1488 correct predictions)
- Confusion between Forward ↔ Left (192 misclassified)  
- Confusion between Forward ↔ Right (49 misclassified)
- Under-representation of Up/Down classes

## Implementation Files Reference

### Training Scripts
- `train_head_pose.py` - PyTorch MobileNetV2 training
- `train_keras_head_pose.py` - Keras multi-architecture training
- `keras_head_pose_model.py` - Model definitions

### Inference
- `inference_head_pose.py` - Production inference with face detection

### Data Processing  
- `preprocess_biwi_dataset.py` - Dataset preprocessing utilities

## Future Improvements Suggested

### Technical Enhancements
1. **Data**: Class balancing, larger datasets (BIWI, 300W-LP, AFLW2000)
2. **Training**: Fine-tuning with partial unfreezing, class-weighted loss
3. **Augmentation**: MixUp/CutMix, stronger geometric augmentations
4. **Architecture**: MobileNetV3, EfficientNetV2-S, lightweight transformers

### Research Directions  
1. **Multi-task learning**: Joint pose + landmarks estimation
2. **Multi-modal**: Depth/IR sensor fusion
3. **Temporal modeling**: Video-based pose tracking
4. **Semi-supervised**: Large unlabeled dataset pretraining

## Citation Format
When citing this work, please reference:
```bibtex
@misc{yourname2024headpose,
  title={Lightweight Head Pose Estimation for Real-Time Applications: MobileNet and Efficient Alternatives},
  author={Your Name},
  year={2024},
  note={Available at: [repository link]}
}
```
