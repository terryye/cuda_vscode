import modal
from modal import FilePatternMatcher

image = (
    modal.Image.from_registry(f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install( ["openmpi-bin", "libopenmpi-dev"])
        .pip_install(["colorama"])
        .env({
            "NCCL_DEBUG": "INFO",
            "OMPI_MCA_btl_vader_single_copy_mechanism": "none", # remove this by add the SYS_PTRACE capability to docker run
        })
        .add_local_dir("util", remote_path="/root/includes/util")
        .add_local_dir("week_07", remote_path="/root/week_07", ignore=FilePatternMatcher("**/output.bin*"))
        .add_local_dir("week_08", remote_path="/root/week_08", ignore=FilePatternMatcher("**/output.bin*"))
        .add_local_file("scripts/monitor_gpu.py", remote_path="/root/monitor_gpu.py")
)

app = modal.App("info7535-skb")

@app.function(image=image, gpu="A100-40gb:1",  timeout=300)
def compile_and_run_cuda_1(code_path: str):
    compile_and_run_cuda(code_path, 1)

@app.function(image=image, gpu="A100-40gb:2",  timeout=300)
def compile_and_run_cuda_2(code_path: str):
    compile_and_run_cuda(code_path, 2)

@app.function(image=image, gpu="A100-40gb:3",  timeout=300)
def compile_and_run_cuda_3(code_path: str):
    compile_and_run_cuda(code_path, 3)

@app.function(image=image, gpu="A100-40gb:4",  timeout=300)
def compile_and_run_cuda_4(code_path: str):
    compile_and_run_cuda(code_path, 4)

@app.function(image=image, gpu="A100-40gb:5",  timeout=300)
def compile_and_run_cuda_5(code_path: str):
    compile_and_run_cuda(code_path, 5)

@app.function(image=image, gpu="A100-40gb:6",  timeout=300)
def compile_and_run_cuda_6(code_path: str):
    compile_and_run_cuda(code_path, 6)

@app.function(image=image, gpu="A100-40gb:7",  timeout=300)
def compile_and_run_cuda_7(code_path: str):
    compile_and_run_cuda(code_path, 7)

@app.function(image=image, gpu="A100-40gb:8",  timeout=300)
def compile_and_run_cuda_8(code_path: str):
    compile_and_run_cuda(code_path, 8)


def copy_c_to_cu(code_path):
    from pathlib import Path
    import os

    source = Path(code_path)
    if source.suffix.lower() == '.c':
        dest = str(source.parent / (source.stem + '.cu'))
        os.rename(source, dest)
        return dest
    return str(source)


def compile_and_run_cuda(code_path: str, cuda_count: int):
    import subprocess
    import threading
    from monitor_gpu import monitor_gpu_utilization

    # Start GPU monitoring in a separate daemon thread
    monitor_thread = threading.Thread(target=monitor_gpu_utilization, args=(5,), daemon=True)
    monitor_thread.start()

    code_path = copy_c_to_cu(code_path)

    # cmd = ["mpicc", "-o", "output.bin", code_path, "-lcudart", "-L/usr/local/cuda/lib64", "-I/usr/local/cuda/include"]
    cmd = ["nvcc", "-DCUDA=1", "-g", "-G", "-rdc=true",
           "-arch=sm_80", # amphere
           "-lmpi",
           "-lstdc++",
           "-lm",
           "-lnccl",
           "-lcudart",
            "-I/root/includes",
            "-I/usr/lib/x86_64-linux-gnu/openmpi/include",  # MPI include path
            "-L/usr/lib/x86_64-linux-gnu/openmpi/lib",      # MPI library path
           "-o", "output.bin", code_path]

    subprocess.run(cmd, text=True,  check=True)

    print("running program")
    subprocess.run([ "mpirun",
                     "--allow-run-as-root", # remove this by fixing the container
                     "-np", str(cuda_count), "./output.bin"], text=True, check=True)
