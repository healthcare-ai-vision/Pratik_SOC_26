# Alzheimer's Disease Biomarker Discovery using ADNI Neuroimaging Data

## Overview

This project focuses on the analysis of structural MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database to identify potential imaging biomarkers associated with Alzheimer's Disease (AD).

The primary objective is to understand neuroimaging data formats, develop a processing pipeline for MRI scans, and investigate structural brain changes that may serve as biomarkers for Alzheimer's Disease.

---

## Project Goals

* Understand the organization of the ADNI database.
* Work with both DICOM (`.dcm`) and NIfTI (`.nii`) neuroimaging formats.
* Load and visualize MRI scans using Python.
* Explore MRI volume characteristics and metadata.
* Build a foundation for future biomarker discovery and machine learning analysis.

---

## Dataset Information

### Source

Alzheimer's Disease Neuroimaging Initiative (ADNI)

### Imaging Modality

* Structural MRI
* T1-weighted MRI (MPRAGE)

### File Formats

#### DICOM (`.dcm`)

DICOM files contain:

* Individual MRI slices
* Scanner metadata
* Acquisition parameters
* Patient information

**Advantages**

* Rich metadata
* Original scanner output

**Challenges**

* Large number of files per scan
* Requires reconstruction into a 3D volume

#### NIfTI (`.nii`)

NIfTI files contain:

* Complete 3D MRI volumes
* Spatial orientation information
* Voxel dimensions

**Advantages**

* Easy loading and visualization
* Standard neuroimaging format
* Suitable for machine learning workflows

**Challenges**

* Less metadata compared to DICOM

---

## Tools and Libraries

The following Python libraries are currently being used:

```python
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
import os
```

---

## Work Completed

### Data Acquisition

* Accessed ADNI neuroimaging datasets.
* Downloaded MRI scans in DICOM and NIfTI formats.
* Organized imaging data for analysis.

### Data Exploration

Examined:

* MRI volume dimensions
* Voxel spacing
* Orientation information
* Metadata contained in DICOM headers

### MRI Visualization

Successfully visualized MRI scans in:

* Axial view
* Coronal view
* Sagittal view

### Neuroimaging Format Analysis

Developed an understanding of:

* DICOM file structure
* NIfTI file structure
* MRI volume representation
* Neuroimaging workflows in Python

---

## Preliminary Observations

Structural MRI scans reveal several brain regions relevant to Alzheimer's Disease research.

Potential biomarkers of interest include:

* Hippocampal atrophy
* Enlargement of lateral ventricles
* Cortical thinning
* Global brain volume reduction
* Regional gray matter loss

These features have been widely reported in Alzheimer's Disease literature and motivate further quantitative analysis.

---

## Challenges Encountered

* Large dataset size.
* Complex ADNI directory structure.
* Multiple scan variants for individual participants.
* Understanding MRI acquisition parameters.
* Handling high-dimensional volumetric imaging data.

---

## Current Status

* ADNI dataset successfully accessed.
* DICOM and NIfTI formats explored.
* MRI visualization pipeline established.
* Initial neuroimaging workflow developed.
* Preliminary investigation of Alzheimer's-related structural changes completed.

---

## Long-Term Objective

The long-term objective of this project is to identify robust neuroimaging biomarkers associated with Alzheimer's Disease and evaluate their potential for disease characterization and early detection using computational analysis and machine learning techniques.
