if __name__ == "__main__":
    import sys
    import time

    import torch

    print(f"python        {sys.version.split()[0]}")
    print(f"torch         {torch.__version__}")
    print(f"cuda build    {torch.version.cuda}")
    print(f"cuda avail    {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        sys.exit("FAIL: no CUDA. Check torch version ends in +cu128, not +cpu.")

    props = torch.cuda.get_device_properties(0)
    print(f"device        {props.name}")
    print(f"capability    sm_{props.major}{props.minor}")
    print(f"vram          {props.total_memory / 1024**3:.1f} GB")
    print(f"arch list     {torch.cuda.get_arch_list()}")

    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    cpu_check = a.cpu() @ b.cpu()
    max_err = (c.cpu() - cpu_check).abs().max().item()
    print(f"matmul        {elapsed * 1000:.1f} ms, max err vs cpu {max_err:.2e}")
    print("OK: GPU works")