import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# LOAD IMAGES
# =====================================================

img1 = cv2.imread("images/glioma.jpg")
img2 = cv2.imread("images/no_tumor.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# =====================================================
# GRAYSCALE
# =====================================================

gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# =====================================================
# RESIZE
# =====================================================

resized1 = cv2.resize(gray1, (256, 256))
resized2 = cv2.resize(gray2, (256, 256))

# =====================================================
# EDGE DETECTION
# =====================================================

edges1 = cv2.Canny(resized1, 100, 200)
edges2 = cv2.Canny(resized2, 100, 200)

# =====================================================
# ADD GAUSSIAN NOISE
# =====================================================

noise1 = np.random.normal(0, 20, resized1.shape)
noise2 = np.random.normal(0, 20, resized2.shape)

noisy1 = resized1 + noise1
noisy2 = resized2 + noise2

noisy1 = np.clip(noisy1, 0, 255).astype(np.uint8)
noisy2 = np.clip(noisy2, 0, 255).astype(np.uint8)

# =====================================================
# DENOISE
# =====================================================

denoised1 = cv2.GaussianBlur(noisy1, (5, 5), 0)
denoised2 = cv2.GaussianBlur(noisy2, (5, 5), 0)

# =====================================================
# DISPLAY EVERYTHING
# =====================================================

fig, ax = plt.subplots(2, 7, figsize=(24, 8))

# ================= IMAGE 1 =================

ax[0,0].imshow(img1)
ax[0,0].set_title("Original")
ax[0,0].axis("off")

ax[0,1].imshow(gray1, cmap="gray")
ax[0,1].set_title("Grayscale")
ax[0,1].axis("off")

ax[0,2].hist(gray1.ravel(), bins=256)
ax[0,2].set_title("Histogram")

ax[0,3].imshow(resized1, cmap="gray")
ax[0,3].set_title("Resized")
ax[0,3].axis("off")

ax[0,4].imshow(edges1, cmap="gray")
ax[0,4].set_title("Edges")
ax[0,4].axis("off")

ax[0,5].imshow(noisy1, cmap="gray")
ax[0,5].set_title("Noisy")
ax[0,5].axis("off")

ax[0,6].imshow(denoised1, cmap="gray")
ax[0,6].set_title("Denoised")
ax[0,6].axis("off")

# ================= IMAGE 2 =================

ax[1,0].imshow(img2)
ax[1,0].set_title("Original")
ax[1,0].axis("off")

ax[1,1].imshow(gray2, cmap="gray")
ax[1,1].set_title("Grayscale")
ax[1,1].axis("off")

ax[1,2].hist(gray2.ravel(), bins=256)
ax[1,2].set_title("Histogram")

ax[1,3].imshow(resized2, cmap="gray")
ax[1,3].set_title("Resized")
ax[1,3].axis("off")

ax[1,4].imshow(edges2, cmap="gray")
ax[1,4].set_title("Edges")
ax[1,4].axis("off")

ax[1,5].imshow(noisy2, cmap="gray")
ax[1,5].set_title("Noisy")
ax[1,5].axis("off")

ax[1,6].imshow(denoised2, cmap="gray")
ax[1,6].set_title("Denoised")
ax[1,6].axis("off")

# =====================================================
# LABEL ROWS
# =====================================================

ax[0,0].set_ylabel("Glioma MRI", fontsize=12)
ax[1,0].set_ylabel("Healthy MRI", fontsize=12)

plt.tight_layout()
plt.show()