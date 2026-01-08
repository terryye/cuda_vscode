# CUDA on macOS – Scaffolding Repo

This repository exists to solve a very specific pain point:

> Mac (including Apple Silicon) has no NVIDIA GPU and no native CUDA support, but many of us still want a **comfortable CUDA development experience** on macOS.

This repo is a **scaffolding / starter kit** that lets you:

-   Write CUDA code comfortably on macOS with **IntelliSense, headers, and syntax highlighting**.
-   Run and verify your code using **Modal (GPU cloud)** or other remote environments.
-   Gradually grow from "just run" to **breakpoint debugging** and eventually **deep profiling**, following the tiered workflow described in the article.

---

## 1. Why this Repo

CUDA is a closed ecosystem with its own extended C++ syntax and toolchain. Compared with Java/Node.js/standard C++, setting up a usable environment—especially on macOS—is not plug-and-play.

This repo packages the **headers, VS Code config, and example projects** I use in my own learning, matching the three-level strategy:

1. **Level 1 – Simple, Free, Runnable**: Mac + VS Code + Modal.com (cloud GPU) – what this repo focuses on most.
2. **Level 2 – Breakpoint Debugging**: Mac + VS Code Remote SSH + KVM VM or WSL2.3.
3. **Level 3 – Deep Profiling**: Windows + Visual Studio + Nsight VSE / Compute.

    If all you want at first is: _“Write CUDA on my Mac in VS Code, hit Run, see `Test Pass`”_ – this repo is for you.

---

## 2. Project Structure

-   `examples/` – CUDA sample programs + helper scripts
    -   `01_simple_kernel/` – Basic single-GPU CUDA kernel, prints threads and `Test Pass!`
    -   `02_mpi/` – CUDA + MPI (multi-process, multi-GPU) example
    -   `03_nccl/` – CUDA + NCCL (collective communication) example
    -   `04_nvshmem/` – CUDA + NVSHMEM (GPU‑shared memory) example
    -   `testall.sh` – Runs all `run_modal_*.sh` scripts, checks for `Test Pass` in output
        -   Green = pass, Red = wrong, plus detailed failure logs
-   `headers/` – Pre‑collected header files from a Linux CUDA environment
    -   `cuda13/` – CUDA runtime / device / math / cuBLAS / cuFFT headers
    -   `nccl/` – NCCL headers
    -   `nvshmem13/` – NVSHMEM headers
    -   `openmpi/` – MPI headers
-   `scripts/` – Modal deployment scripts (Docker-based GPU cloud)
    -   `modal_nvcc.py` – Compile & run a single‑GPU CUDA file on Modal
    -   `modal_mpi.py` – Multi‑GPU MPI CUDA (with GPU utilization monitoring)
    -   `modal_nccl.py` – NCCL examples (if present)
    -   `modal_nvshmem.py` – NVSHMEM examples on H100 (NVSHMEM build baked into image)
    -   `install.sh` – One‑click CUDA + tools installer for remote GPU servers / VMs

---

## 3. Level 1 – Mac + VS Code + Modal (Recommended Starting Point)

**Goal:** On macOS, get **good editor experience** (no red squiggles, proper autocomplete) and a **one‑command way** to run CUDA code in the cloud.

### 3.1 Prerequisites

-   macOS with VS Code installed
-   VS Code extensions:
    -   Nsight Visual Studio Code Edition
    -   Remote SSH (for later Level 2, optional)
-   Python 3.11+ (for Modal CLI)
-   Modal account and CLI:
    ```bash
    pip install modal
    modal token set
    ```

### 3.2 Clone This Repo

```bash
git clone https://github.com/terryye/cuda_vscode.git
cd cuda_vscode
```

Open the folder in VS Code.

### 3.3 VS Code IntelliSense on macOS

macOS has no CUDA toolkit installed, so VS Code can’t find headers like `cuda_runtime.h` by default. This repo solves it by **vendor‑shipping** the needed headers in `headers/` and wiring them into VS Code’s C/C++ config.

Check / adjust:

-   `.vscode/c_cpp_properties.json` – includes search paths under `headers/` for:
    -   CUDA
    -   MPI
    -   NCCL
    -   NVSHMEM

Result: `.cu` files get proper **syntax highlighting, symbol resolution, and autocomplete** on macOS, even with constructs like `<<<1, 4>>>` and APIs like `cudaDeviceSynchronize`.

---

## 4. Running the Examples

### 4.1 Run on Modal (Cloud GPUs)

Each example folder has a `run_modal_*.sh` script that:

-   Calls the corresponding `scripts/modal_*.py` Modal app
-   Sends `./main.cu` as `--code-path`
-   Compiles and runs on a remote GPU (H100 / A100, depending on script)

Example:

```bash
cd examples/01_simple_kernel
./run_modal_nvcc.sh
```

You should see the kernel output and finally `Test Pass!`.

For MPI / NCCL / NVSHMEM:

```bash
cd examples/02_mpi
./run_modal_mpi.sh

cd ../03_nccl
./run_modal_mpi.sh     # or matching script name

cd ../04_nvshmem
./run_modal_nvshmem.sh
```

Some scripts accept environment variables such as `N_GPU` that control how many GPUs / ranks to use.

### 4.2 Run Local (If You Have a PC + GPU)

If you also have a local CUDA machine (Linux / Windows + WSL2), you can copy / clone this repo there and use the `run_local_*.sh` scripts under `examples/` (requires CUDA toolkit + drivers installed). This corresponds to the **“Local PC + GPU”** option from the article.

### 4.3 Batch Test All Modal Examples

From the repo root:

```bash
cd examples
./testall.sh
```

The script will:

-   Detect OS (prints a yellow warning on macOS / Windows)
-   For each subfolder, run the first `run_modal_*.sh`
-   Search the output for `Test Pass`
-   Print results:
    -   Green `pass` if found
    -   Red `wrong` otherwise, with full captured output

---

## 5. Beyond Level 1 – How This Repo Fits Your Tiered Strategy

This repo mainly delivers **Level 1** (Mac + VS Code + Modal). You can also reuse it when moving up the stack:

### Level 2 – Breakpoint Debugging (Mac + Remote VM / WSL2)

Use the same codebase, but run it on an environment that supports `cuda-gdb`:

1. **KVM VM (e.g., Vast.ai)**

    - Filter for images tagged with `VM` in Vast.ai.
    - SSH in and clone this repo.
    - Run `scripts/install.sh` to install the necessary CUDA toolkit, MPI, Nsight support, and related tools on the remote server.
    - Use VS Code Remote SSH for coding and breakpoint debugging.

2. **Local PC + WSL2 (Ubuntu)**
    - Open WSL2 Terminal and clone this repo.
    - Run `scripts/install.sh` to install the necessary CUDA toolkit, MPI, Nsight support, and related tools on the remote server.
    - open WSL2 filesystem in Windws VS Code (Remote – WSL)
      or enable ssh for WSL2 and Use Windws/Mac VS Code Remote SSH for coding and breakpoint debugging.

### Level 3 – Deep Optimization (Windows + Visual Studio + Nsight)

When you need **profiling, warp‑level analysis, and timeline views**, VS Code’s UI becomes limiting. At that point:

-   Use a Windows machine with an NVIDIA GPU.
-   Install **Visual Studio 2022 + Nsight VSE + Nsight Compute**.
-   Port your own code into a Visual Studio CUDA project.

This gives you the “king of debugging” experience.

---

## 6. Hardware / Cloud Notes

-   **Docker‑based GPU clouds (e.g., Modal.com, Most of the GPU clouds)**

    -   Pros: Cheap, fast start, generous free tiers.
    -   Limitation: Typically no `SYS_PTRACE` → no `cuda-gdb` breakpoints.
    -   Best for: Running and validating logic (printf‑style debugging).

-   **KVM VMs (e.g., Vast.ai with VM images)**

    -   Pros: Supports `cuda-gdb` breakpoints; very cheap per hour.
    -   Cons: No deep profiling; instances are not exclusive; persistence/queueing can be clunky.

-   **Bare‑metal servers (e.g., Voltage Park)**

    -   Pros: Full control, all tools work (including profilers).
    -   Cons: Expensive and often in short supply.

-   **Local PC + GTX 1650 (recommended for learners)**
    -   Fully supported by modern CUDA.
    -   Excellent price/performance and enough for most learning tasks.

For multi‑GPU (NVLink) experiments, prefer two RTX 20xx/30xx cards with real NVLink. Consumer RTX 40xx removed hardware NVLink and future consumer lines may deprecate even driver‑level support.

---

## 7. My Suggested Workflow with This Repo

-   Do **90% of coding on Mac in VS Code** using this repo for headers + examples.
-   Use **Modal** via the provided scripts to run and validate code quickly.
-   When you need breakpoints, either:
    -   Connect from Mac VS Code to your own PC (Windows + WSL2) via Remote SSH, or
    -   Rent a short‑lived **KVM VM** (e.g., on Vast.ai), clone this repo there, and debug.
-   For rare cases needing deep profiling / visualization, switch to a Windows box with Visual Studio + Nsight.

This way you get a comfortable macOS editing experience, cheap cloud GPUs for quick runs, and an upgrade path to serious debugging when you need it.

---

## 8. Notes & Contributions

-   Feel free to add more examples (e.g., cuBLAS/cuFFT/Thrust) following the existing folder pattern.
-   If you improve the VS Code configuration for macOS CUDA development, PRs are welcome.

Happy CUDA hacking on macOS!
