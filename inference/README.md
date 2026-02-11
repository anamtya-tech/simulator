**For Ubuntu 24.04 on x86_64 (Azure VM):**

**1. Install dependencies:**
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential wget unzip nlohmann-json3-dev
```

**2. Download LibTorch for Linux CPU:**
```bash
cd ~
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip
```

**3. Create project directory:**
```bash
mkdir ~/simulator/inference && cd ~/simulator/inference
```

**4: create cpp and makefile**

**5. Build:**
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
cmake --build . --config Release
```

**6. Run:**
```bash
./inference
```

This should work on your Ubuntu 24.04 VM. Any errors during build?