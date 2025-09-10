import modal

image = (
    modal.Image.from_registry(f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
        .env({
            "NCCL_DEBUG": "INFO",
        })
        .add_local_dir(".", remote_path="/hpc-for-ai")
)

app = modal.App("llama2.cu")

@app.function(image=image, gpu="A100-40gb", timeout=300)
def compile_and_run_cuda(code_path: str):
    import subprocess

    subprocess.run(["nvcc", "-DCUDA=1", "-g", "-G", "-rdc=true", "-arch=native", code_path, "-o", "output.bin"],
                   text=True,  check=True)
    subprocess.run([ "./output.bin"], text=True, check=True)