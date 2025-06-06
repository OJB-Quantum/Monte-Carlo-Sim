# Monte-Carlo-Sim
Some interesting examples of Monte Carlo simulations performed with [CUDA Python](https://nvidia.github.io/cuda-python/cuda-core/latest/index.html)/[CuPy](https://docs.cupy.dev/en/stable/overview.html) in Google Colab. The notebooks are authored by Onri Jay Benally with citations, if relevant.

No need to download anything manually. Just run the notebooks.

### Click here to render the notebooks in the browser: [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/OJB-Quantum/Monte-Carlo-Sim/tree/main/)

---

### Below are some examples pulled from the Colab notebooks:


> Josephson Junction Quantum Tunneling Prediction

![Untitled](https://github.com/user-attachments/assets/5a763407-c31f-4b75-924f-6db97bbc39b8)

---

> 3D Ion Beam Etching Simulation

![image](https://github.com/user-attachments/assets/bb88bc0b-b620-4e35-b666-704aa6d5e75e)

---

> 2-D Heat Equation

![Untitled](https://github.com/user-attachments/assets/881ad174-2794-41ea-941b-1bd501f5b525)

![Untitled](https://github.com/user-attachments/assets/76dbca80-6f21-4f3c-b637-8c6e6f9ed58f)

---

> Egg White Resist Electron-Beam Penetration Simulation

![Untitled](https://github.com/user-attachments/assets/aebcac31-aa04-40ae-b6f3-b3da7476f32f)

![Untitled](https://github.com/user-attachments/assets/7823b8db-3aeb-4e15-aeaf-108cfbfecd6b)

---

> Terabyte-Level L1 Cache Prediction

![Untitled](https://github.com/user-attachments/assets/c8250abc-e625-4540-a7a3-7367464670ea)
![Untitled](https://github.com/user-attachments/assets/d1580a58-e483-41f6-b93c-9c2bd863878e)
![image](https://github.com/user-attachments/assets/46df41c8-0bc2-481f-ab9a-5236c4f37c74)

---

> Semantic Shift Simulation

![Untitled](https://github.com/user-attachments/assets/d732b223-b26d-41ec-b703-d342dfb3eb4d)
![Untitled](https://github.com/user-attachments/assets/b1df8b4d-54c2-4079-b074-90876054d140)

---

# GPU Library Comparison Cheat‑Sheet for Google Colab

This table summarizes practical differences when using **CuPy**, **CUDA Python** (Numba / NVIDIA `cuda‑python`) or **Julia CUDA** on Colab’s three main GPU options.

## Free GPU Tier (Tesla T4, 16 GB)

| Aspect                        | CuPy                                                                                      | CUDA Python (Numba / `cuda-python`)                                         | Julia CUDA                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Pre‑installed?**            | Yes `cupy‑cuda12x ≥ 13.3` already in Colab base image                                     | Yes Numba present, but PTX version may lag driver                           | No Julia kernel optional; add with `Pkg.add("CUDA")`      |
| **One‑liner setup**           | *(usually none)*<br>`# upgrade only if needed`<br>`pip install -q --upgrade cupy-cuda12x` | `pip install -q --upgrade "numba[cuda]"`<br>plus env vars if NVVM not found | `using Pkg; Pkg.add("CUDA")`                              |
| **Kernel authoring style**    | NumPy‑like array ops; optional `RawKernel` / `cupyx.jit` for custom GPU code              | Full control: `@cuda.jit` on Python or embed PTX/CUDA C strings             | Full control: `@cuda` kernels in Julia                    |
| **Library coverage**          | cuBLAS, cuFFT, cuSOLVER, cuSPARSE, NCCL, cuDNN                                            | You invoke CUDA libs manually or via `numba.cuda` driver calls              | Julia wrappers for BLAS/FFT/DNN; high‑level `CuArray` API |
| **Typical speed‑up vs NumPy** | 20–60× for vectorized math                                                                | Similar if kernels tuned; launch overhead on tiny arrays                    | Comparable; sometimes +5 % from LLVM optimizations        |
| **Common pitfalls**           | Duplicate wheels (11× & 12×) break loader; OOM at 16 GB                                   | `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` after Colab image update               | First run pre‑compiles packages (30–60 s)                 |
| **Best‑fit workloads**        | Drop‑in acceleration for array algebra, FFTs, ML inference                                | Custom Monte‑Carlo, stencils, irregular memory access                       | Native Julia data/ML pipelines needing GPU                |

---

## Paid GPU — NVIDIA **L4** (Ada Lovelace, CC 8.9, 24 GB)

| Aspect                    | CuPy                                         | CUDA Python                                       | Julia CUDA                      |
| ------------------------- | -------------------------------------------- | ------------------------------------------------- | ------------------------------- |
| **Wheel / package**       | `cupy‑cuda12x ≥ 13.3` ships fatbin for SM 89 | **Numba ≥ 0.61** required for SM 89 PTX           | `CUDA.jl` auto‑detects arch     |
| **Precision extras**      | FP8 tensor‑core matmul via `precision="fp8"` | Need inline PTX / CUTLASS kernels for FP8         | `allow_fp8!()` (CUDA.jl 5.1+)   |
| **Memory mgmt**           | Pool hides `cudaMalloc`; 24 GB ceiling       | Manual or managed; same ceiling                   | Automatic through Julia runtime |
| **Perf vs T4**            | \~3 × on dense matmul / conv                 | Similar once tuned; fewer SMs can limit occupancy | Similar to CuPy                 |
| **Limitations**           | BW ≈ 300 GB/s (PCIe 4.0), not HBM            | Same bandwidth cap                                | Same                            |
| **Colab cost (Pro/Pro+)** | ≈ \$0.48 hr⁻¹ (4.8 CU hr⁻¹)                  | idem                                              | idem                            |

---

## Paid GPU — NVIDIA **A100 40 GB** (Ampere, CC 8.0, HBM2e)

| Aspect               | CuPy                                                | CUDA Python                                     | Julia CUDA                   |
| -------------------- | --------------------------------------------------- | ----------------------------------------------- | ---------------------------- |
| **Wheel / package**  | Same `cupy‑cuda12x` covers SM 80                    | **Numba ≥ 0.57** supports SM 80                 | `CUDA.jl` auto               |
| **Precision extras** | Enable TF32: `cp.cuda.set_matmul_precision("tf32")` | `@cuda.jit(fastmath=True)` -> TF32 tensor cores | `allow_tf32!()`              |
| **Memory / BW**      | 40 GB HBM2e, 1.6 TB s⁻¹                             | same                                            | same                         |
| **Perf gain vs T4**  | 10–15 × on GEMM / conv                              | Similar after occupancy tuning                  | Similar                      |
| **Session cost**     | ≈ \$1.18 hr⁻¹ (11.8 CU hr⁻¹)                        | idem                                            | idem                         |
| **Caveats**          | Limited availability; CU burn fast                  | Kernel launch + compile time higher             | Initial package compile time |

---

### Quick Recommendations

* **Start with CuPy** for anything expressible as NumPy/SciPy—lowest friction, high speed.
* **Use CUDA Python** only for the hotspots that need bespoke parallel patterns; stay on the latest Numba.
* **Prefer Julia CUDA** if your workflow is already in Julia—performance parity with cleaner syntax.
* **Choose GPU by memory and budget**: Free T4 for prototyping; L4 for moderate models with FP8; A100 when you need 40 GB or TF32 accuracy.

> Yes = works out‑of‑the‑box No = requires explicit install / setup
