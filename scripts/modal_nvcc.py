import modal
from modal import FilePatternMatcher
import os
import glob

image = (
    modal.Image.from_registry(f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
        .env({
            "NCCL_DEBUG": "INFO",
        })
        .add_local_dir("util", remote_path="/root/includes/util")
)

# Dynamically add all week_* folders
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
week_folders = sorted(glob.glob(os.path.join(project_root, "week_*")))

for week_folder in week_folders:
    if os.path.isdir(week_folder):
        folder_name = os.path.basename(week_folder)
        image = image.add_local_dir(
            folder_name,
            remote_path=f"/root/{folder_name}",
            ignore=FilePatternMatcher("**/output.bin*")
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