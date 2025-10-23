

def monitor_gpu_utilization(interval=5):
    import subprocess
    import time
    from colorama import Fore, Style, init
    from datetime import timedelta

    """Logs GPU utilization every `interval` seconds."""
    try:
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            wall_time = str(timedelta(seconds=elapsed)).split('.')[0] + f".{int(elapsed % 1 * 10):01d}"

            command = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"
            ]
            output = subprocess.check_output(command, encoding="utf-8")

            print("\n" + "=" * 60)
            print(f"{Fore.CYAN}GPU STATUS [Wall time: {wall_time}]")
            print("=" * 60)

            for line in output.strip().splitlines():
                idx, name, mem_used, mem_total, util = [x.strip() for x in line.split(",")]

                print(f"{Fore.CYAN}GPU {idx} {Fore.MAGENTA}({name}): "
                      f"{Fore.GREEN}{mem_used}/{mem_total} MiB, "
                      f"{Fore.YELLOW}{util}% utilization")

            time.sleep(interval)

    except Exception as e:
        print(f"GPU monitoring stopped: {e}")