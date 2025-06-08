#  CUDA Array Summation: CPU vs GPU Performance

This project demonstrates the performance comparison between a **CPU-based** and a **CUDA GPU-based** approach to summing large arrays.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/AnaghaSujith/cuda-array-sum/blob/main/cuda-array-sum.ipynb)


---

## Overview

We sum a 40 million element array where every element is `1`:

- **GPU**: Parallel reduction using shared memory and `atomicAdd`
- **CPU**: Sequential summation using a `for` loop
- **Goal**: Compare runtime performance between both

---

## Files

| File                     | Description                                      |
|--------------------------|--------------------------------------------------|
| `cuda_array_sum.ipynb`   | Main CUDA-enabled Jupyter notebook (Colab)       |
| `array_sum2.cu`          | Standalone CUDA file (optional)                  |
| `sample_output.txt`      | Sample output showing results & timings          |
| `benchmark_results.csv`  | Test results for different array sizes           |
| `assets/performance_chart.png` | Bar chart comparing GPU vs CPU timings     |

---

## 📊 Sample Benchmark

| Array Size | CPU Time (s) | GPU Time (s) |
|------------|--------------|--------------|
| 10M        | 0.025877     | 0.001019     |
| 20M        | 0.052833     | 0.001932     |
| 40M        | 0.109267     | 0.003739     |

 *See `assets/performance_chart.png` for graphical view.*

---

## Technologies Used

- CUDA (via Colab)
- Python & Jupyter Notebook
- C++
- Shared memory & parallel reduction
- Google Colab environment for GPU runtime

---

## ▶️ How to Run on Colab

1. Click the badge above ☝️
2. Make sure GPU is enabled:  
   `Runtime > Change runtime type > GPU`
3. Run all cells

---

## How to Run `.cu` File Locally

```bash
nvcc array_sum2.cu -o array_sum
./array_sum
