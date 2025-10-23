import modal

# Create a custom image with SSH server
ssh_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04")
    .apt_install(
        "openssh-server", 
        "sudo", 
        "curl", 
        "wget", 
        "git", 
        "vim",
        "build-essential",
        "cmake"
    )
    .pip_install("numpy", "torch", "ipython")
    .run_commands(
        # Configure SSH
        "mkdir -p /var/run/sshd",
        "echo 'root:modaldev' | chpasswd",  # Set a password
        "sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config",
        "sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config",
        "echo 'AllowUsers root' >> /etc/ssh/sshd_config",
    )
)

app = modal.App("cuda-dev-ssh")


@app.function(
    image=ssh_image,
    gpu="T4",  # or "A10G", "A100", etc.
    cpu=4,
    memory=16384,  # 16GB RAM
    timeout=3600 * 4,  # 4 hours
    volumes={
        "/workspace": modal.Volume.from_name("cuda-workspace", create_if_missing=True)
    }
)
def dev_container():
    import subprocess
    import socket
    import time
    
    # Start SSH service
    subprocess.run(["/usr/sbin/sshd", "-D", "&"], shell=True, check=False)
    
    # Get the container's hostname
    hostname = socket.gethostname()
    print(f"Container started with hostname: {hostname}")
    print("SSH server is running...")
    print("=" * 50)
    print("To connect via SSH:")
    print(f"1. Get the container ID: modal container list")
    print(f"2. Create SSH tunnel: modal container exec [container-id] --no-pty nc -l 2222")
    print(f"3. In another terminal: ssh -p 2222 root@localhost")
    print("Password: modaldev")
    print("=" * 50)
    
    # Keep container alive
    while True:
        time.sleep(60)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Container still running...")