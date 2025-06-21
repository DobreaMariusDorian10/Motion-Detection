# 🎥 Background Subtraction and Motion Detection from Video using OpenCV

This project performs **background estimation** and **motion detection** from a video file (`videoclipmasini.mp4`) using OpenCV in Python. It uses both **median** and **mean** (average) background modeling techniques and applies several thresholding methods (fixed threshold, adaptive, and Otsu) to detect motion in video frames.

---

## 📸 Input & Output Preview

### 🎞 Original Video (GIF Preview)
![Original Video](https://github.com/DobreaMariusDorian10/Motion-Detection/blob/78ba63374486ceb519929bc120af24ed3aa656a3/videoclipmasini.gif)

### 🎯 Motion Detection Result (Median + Otsu)
![Motion Detection Output](https://github.com/DobreaMariusDorian10/Motion-Detection/blob/78ba63374486ceb519929bc120af24ed3aa656a3/video_dif_mediana_otsu.gif)

---

## 📌 Features

- ✅ Extracts evenly spaced frames from video  
- ✅ Builds two types of background images:
  - Median background
  - Mean (average) background
- ✅ Compares each frame to the background
- ✅ Applies different thresholding methods to highlight moving objects:
  - Fixed threshold
  - Adaptive threshold
  - Otsu's method
- ✅ Saves six processed videos showing detected motion:
  - Median + Fixed Threshold
  - Median + Adaptive Threshold
  - Median + Otsu
  - Mean + Fixed Threshold
  - Mean + Adaptive Threshold
  - Mean + Otsu

---

## 🧠 Techniques Used

- **Background Estimation**:  
  - `np.median()` for static background modeling  
  - `np.mean()` for average-based background modeling  
- **Noise Reduction**:  
  - Gaussian blurring to reduce noise before thresholding
- **Thresholding**:
  - `cv2.threshold()` with fixed value
  - `cv2.adaptiveThreshold()` for local adaptation
  - `cv2.threshold(..., + cv2.THRESH_OTSU)` for dynamic thresholding

---

## 🔧 Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib (for displaying frames)
- Google Colab (for `cv2_imshow()` and file operations)

### Install required libraries:
```bash
pip install opencv-python numpy matplotlib
