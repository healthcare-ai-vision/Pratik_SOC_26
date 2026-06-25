# Model Diagnostic Report: Brain Scan Classification

## 1. Executive Summary

This report summarizes the performance evaluation and explainability analysis of the current YOLOv11-based brain scan classification model. While the model demonstrates strong learning convergence, error analysis highlights specific vulnerabilities in distinguishing between subtle impairment levels.

---

## 2. Training Performance

### Convergence
The training and validation loss curves indicate a well-fitted model, showing minimal signs of overfitting after 50 epochs.

### Accuracy
The model reached a stable performance plateau and effectively distinguishes between the primary diagnostic classes.

---

## 3. Error Analysis

A total of **47 misclassifications** were identified in the test dataset. Analysis of these errors reveals recurring confusion patterns:

| Confusion Type | Frequency | Clinical Implications |
|---------------|-----------|----------------------|
| Mild Impairment ↔ No Impairment | Moderate | Suggests difficulty in identifying subtle early-stage atrophy. |
| No Impairment ↔ Very Mild | High | Indicates overlap between normal and early abnormal categories. |
| Very Mild ↔ Mild Impairment | Moderate | Highlights challenges in fine-grained feature discrimination. |

### Key Observations

- Most errors occur between adjacent disease stages.
- The model performs well on clearly distinguishable classes.
- Decision boundaries between normal aging and early neurodegeneration remain difficult to separate.

---

## 4. Explainability Findings (Grad-CAM)

Grad-CAM was used to visualize the regions influencing model predictions.

### Successful Activations

For correctly classified samples, the model consistently focused on:

- Ventricular regions
- Hippocampal structures

These regions are anatomically associated with neurodegenerative progression and are clinically relevant biomarkers for Alzheimer's disease.

### Failure Modes

Several misclassified images produced nearly uniform or featureless Grad-CAM heatmaps.

This behavior suggests **model uncertainty**, where:

1. The image lacks sufficiently discriminative features.
2. The network cannot confidently identify disease-specific patterns.
3. The prediction defaults toward the **"No Impairment"** class.

Example indicators of uncertainty include:

- Flat activation maps
- Weak localized attention
- Low contrast between highlighted and non-highlighted regions

---

## 5. Strategic Recommendations

### Immediate Technical Actions

#### Data Curation

Expand the training dataset for:

- Very Mild Impairment
- Mild Impairment

This will help improve class separation and sharpen decision boundaries.

#### Target Layer Optimization

Current Grad-CAM visualizations are generated from the final network layer (`[-2]`).

Consider targeting intermediate convolutional layers such as:

- `[-4]`
- `[-5]`

These layers often capture more detailed texture and structural information, leading to more informative explanations.

#### Image Preprocessing

Implement **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance subtle anatomical variations that may currently be overlooked.

Potential benefits:

- Improved local contrast
- Better visibility of structural degeneration
- Enhanced feature extraction

---

## 6. Future Development Directions

### Anatomical Masking

Introduce automated preprocessing techniques such as:

- Skull stripping
- Brain extraction
- Region-of-interest cropping

Benefits include:

- Removal of irrelevant background information
- Reduced noise from image borders
- Increased focus on clinically meaningful brain structures

### Transition to 3D Analysis

The current approach relies on 2D slice classification, which does not fully utilize volumetric information.

Future work should investigate:

- 3D Convolutional Neural Networks (3D CNNs)
- Volumetric MRI processing using NIfTI files
- Libraries such as `nibabel` for 3D medical image handling

Expected advantages:

- Preservation of spatial context across slices
- Improved representation of anatomical structures
- Enhanced diagnostic performance for neurological disorders

---

## Conclusion

The YOLOv11-based classifier demonstrates strong convergence and promising classification capability. However, performance degradation occurs when distinguishing between neighboring impairment stages, particularly in early disease progression.

Grad-CAM analysis confirms that the model learns clinically relevant anatomical features but also reveals uncertainty-driven failure modes. Future improvements should focus on:

1. Expanding underrepresented classes.
2. Optimizing explainability layers.
3. Enhancing image preprocessing.
4. Incorporating anatomical masking.
5. Transitioning toward volumetric (3D) deep learning approaches.

These improvements are expected to increase both predictive accuracy and clinical interpretability.
Some results are attached in some_results and error_diagnostics directories.
