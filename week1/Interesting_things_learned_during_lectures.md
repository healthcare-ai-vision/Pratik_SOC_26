# Interesting Things Learned During Lectures

### 1. YOLO (You Only Look Once)
Unlike traditional object detection models that scan an image multiple times using sliding windows, YOLO processes the entire image in a single forward pass through the neural network. By dividing the image into a grid and predicting bounding boxes and class probabilities simultaneously, it achieves incredible processing speeds. This makes YOLO the gold standard for real-time AI applications like autonomous driving and live surveillance tracking.

### 2. Stereo Vision (Stereos)
Stereo vision mimics human depth perception by processing two slightly offset images captured by dual cameras to calculate the distance of objects. By analyzing the "disparity"—the relative shift of pixels between the left and right images—AI models can construct highly accurate 3D depth maps of an environment. This technique is vital in robotics and spatial AI, allowing machines to navigate complex physical spaces safely without relying solely on active sensors like LiDAR.

### 3. Noising an Image
While adding random noise (like Gaussian or salt-and-pepper noise) to a clean image seems counterintuitive, it is a powerful technique used to train robust AI models. Introducing controlled noise prevents machine learning models from overfitting to perfect dataset conditions, forcing them to learn the underlying structural features instead. Furthermore, mastering noise is essential for training diffusion models and image-denoising autoencoders that learn how to clean up corrupted medical or satellite imagery.

### 4. Image Scaling
Image scaling involves resizing an image by interpolating pixels, which is a crucial preprocessing step since most deep learning architectures require inputs to have fixed, uniform dimensions. Downscaling reduces the computational load and training time, while upscaling must be handled carefully using advanced interpolation techniques to avoid losing critical edge details. In modern AI, deep learning models are also used for "Super-Resolution," where the network intelligently predicts and fills in missing pixels to enhance image quality.

### 5. Image Cropping
Cropping is a foundational preprocessing technique used to isolate a specific Region of Interest (ROI) from a larger image, removing irrelevant background clutter that might confuse an AI model. In object detection workflows, cropping helps extract bounding-box data to build clean classification datasets. It is also frequently utilized in "Data Augmentation," where random cropping forces the neural network to recognize objects even when they are partially visible or positioned at the edges of the frame.