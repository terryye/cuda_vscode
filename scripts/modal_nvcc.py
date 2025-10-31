import modal
from modal import FilePatternMatcher
import os

image = (
    modal.Image.from_registry(f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
        .env({
            "NCCL_DEBUG": "INFO",
        })
        .add_local_dir("util", remote_path="/root/includes/util")
        .add_local_dir("week_01", remote_path="/root/week_01", ignore=FilePatternMatcher("**/output.bin*"))
        .add_local_dir("week_02", remote_path="/root/week_02", ignore=FilePatternMatcher("**/output.bin*"))
        .add_local_dir("week_03", remote_path="/root/week_03", ignore=FilePatternMatcher("**/output.bin*"))
        .add_local_dir("week_04", remote_path="/root/week_04", ignore=FilePatternMatcher("**/output.bin*"))
        .add_local_dir("week_05", remote_path="/root/week_05", ignore=FilePatternMatcher("**/output.bin*"))
        .add_local_dir("week_07", remote_path="/root/week_07", ignore=FilePatternMatcher("**/output.bin*"))

)
app = modal.App("nvcc")

@app.function(image=image, gpu="A100-40gb", timeout=300)
def compile_and_run_cuda(code_path: str):
    import subprocess

    #subprocess.run(["find", "."])
    #subprocess.run(["pwd"])

    subprocess.run(["nvcc", "-DCUDA=1", "-g", "-G", "-rdc=true", "-arch=native", 
                    "-I/root/includes/",
                    code_path, "-o", "output.bin"],
                   text=True,  check=True)
    subprocess.run([ "./output.bin"], text=True, check=True)