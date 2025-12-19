import modal
import os


script_dir = os.path.dirname(__file__)

NDEVICES = int(os.environ.get('N_GPU', 2))

NVSHMEM_VERSION = "3.4.5-0"
NVSHMEM_PREFIX = "/opt/nvshmem"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install(
        [
            "wget",
            "build-essential",
            "cmake",
            "libhwloc-dev",
            "libnuma-dev",
            "openmpi-bin",
            "openmpi-common",
            "libopenmpi-dev"
        ]
    )
    .run_commands(
        [
            (
                "set -eux; "
                "cd /tmp && "
                f"wget -O nvshmem-{NVSHMEM_VERSION}.tar.gz "
                f"https://github.com/NVIDIA/nvshmem/archive/refs/tags/v{NVSHMEM_VERSION}.tar.gz && "
                f"tar xvf nvshmem-{NVSHMEM_VERSION}.tar.gz && "
                f"cd nvshmem-{NVSHMEM_VERSION} && "
                # disable IBRC/UCX via env so IB (verbs.h) is never needed
                "export CUDA_HOME=/usr/local/cuda "
                "NVSHMEM_IBRC_SUPPORT=0 "
                "NVSHMEM_UCX_SUPPORT=0; "
                # configure
                "cmake -S . -B build "
                f"-DCMAKE_BUILD_TYPE=Release "
                f"-DCMAKE_INSTALL_PREFIX={NVSHMEM_PREFIX} "
                f"-DNVSHMEM_PREFIX={NVSHMEM_PREFIX} "
                f"-DNVSHMEM_MPI_SUPPORT=1 "
                f"-DNVSHMEM_SHMEM_SUPPORT=0 "
                f"-DNVSHMEM_UCX_SUPPORT=0 "
                f"-DNVSHMEM_LIBFABRIC_SUPPORT=0 "
                f"-DNVSHMEM_IBRC_SUPPORT=0 "
                f"-DNVSHMEM_BUILD_TESTS=0 "
                f"-DNVSHMEM_BUILD_EXAMPLES=0 "
                f"-DNVSHMEM_BUILD_PACKAGES=0 "
                f"-DNVSHMEM_BUILD_PYTHON_LIB=OFF "
                f"-DCUDA_ARCHITECTURES=90 && "
                # build and install
                "cmake --build build -j && "
                "cmake --install build"
            ),
        ]
    )
    .add_local_dir(".", remote_path="/root")
    .add_local_dir(script_dir, remote_path="/root/scripts")
)

app = modal.App("nvshmem2")


@app.function(gpu=f"H100:{NDEVICES}", image=image)
def run(code_path: str):
    import os
    import subprocess

    nvshmem_home = os.environ.get("NVSHMEM_HOME", NVSHMEM_PREFIX)

    os.environ["NVSHMEM_HOME"] = nvshmem_home
    os.environ["LD_LIBRARY_PATH"] = (
        f"{nvshmem_home}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"  # often needed too
    os.environ["PMIX_MCA_gds"] = "hash"
    os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"

    compile_cmd = [
        "nvcc",
        "-rdc=true",
        "-ccbin",
        "mpicxx",
        "-arch=sm_90",  # H100
        "-I",
        f"{nvshmem_home}/include",
        code_path,
        "-o",
        "output",
        "-L",
        f"{nvshmem_home}/lib",
        "-lnvshmem_host",
        "-lnvshmem_device",
        "-lcudart",
        "-lcuda",
        "-lnvidia-ml",
        "-lmpi",
        "-lstdc++",
        "-lm",
        "-lnccl",
        "-DCUDA=1"
    ]

    print("Compile command:")
    print(" ".join(compile_cmd))

    subprocess.run(compile_cmd, check=True)
    run_cmd = [
        "mpirun",
        "-mca", "btl_vader_single_copy_mechanism", "none",
        "-np", f"{NDEVICES}",   # number of PEs (processes)
        "./output",
    ]

    print("Run command:")
    print(" ".join(run_cmd))
    subprocess.run(run_cmd, check=True)
