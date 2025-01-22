# TensorFlow Deep Learning Project

This project uses TensorFlow with GPU acceleration to train deep learning models efficiently on an NVIDIA RTX 3060 Laptop GPU.

## **Installation Instructions**

Follow these steps to set up the environment and run the project.

### **1. Prerequisites**

- Ensure you have an **NVIDIA GPU** with at least 6GB VRAM.
- Install **CUDA 12.2** and **cuDNN 8.6** (Check with `nvidia-smi`).
- Install **Anaconda** (Recommended).

### **2. Setting Up the Environment**

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/deep-learning-project.git
    cd deep-learning-project
    ```

2. Create and activate a new Conda environment:

    ```bash
    conda create --name dl-env python=3.12 -y
    conda activate dl-env
    ```

3. Install dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### **3. Running the Project**

1. Ensure your GPU is available:

    ```bash
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

2. Run the training script:

    ```bash
    python train.py
    ```

### **4. GPU Memory Optimization**

If you encounter **out-of-memory (OOM)** errors, follow these optimizations:

- **Enable TensorFlow memory growth:**

    Add the following snippet in your Python script:

    ```python
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    ```

- **Reduce batch size** to avoid memory overflow:

    ```python
    model.fit(x_train, y_train, batch_size=8, epochs=25)
    ```

- **Enable mixed precision training** to reduce memory usage:

    ```python
    from tensorflow.keras import mixed_precision

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    ```

- **Clear memory after training:**

    ```python
    import gc
    import tensorflow as tf

    tf.keras.backend.clear_session()
    gc.collect()
    ```

### **5. Common Issues and Fixes**

#### **1. Out of Memory (OOM) Error**
- Kill other running processes using GPU:
    ```bash
    nvidia-smi
    kill -9 <PID>
    ```
- Reduce batch size in training.

#### **2. TensorFlow GPU Not Detected**
- Ensure CUDA and cuDNN paths are set correctly.
- Run the following command to check GPU:

    ```bash
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

#### **3. Library Compatibility Issues**
If you encounter errors related to incompatible library versions, try downgrading or reinstalling dependencies:

    ```bash
    pip install --upgrade tensorflow-gpu keras numpy<2
    ```

### **6. Additional Resources**
- [TensorFlow GPU Installation Guide](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Keras Documentation](https://keras.io/)



