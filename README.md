# CPP CUDA VSCODE Debug DEMO

## Overview

This project demonstrates the debugging and development of CUDA kernels using C++ and the Nsight Visual Studio Code Edition. It utilizes CMake for building the project and targets NVIDIA Turing architecture (compute capability 7.5). The project includes a scatter operation implementation and tests for point cloud data processing.

To step into CUDA kernel functions during debugging, you must use `"type": "cuda-gdb"` in the `launch.json` file under the `.vscode` folder.

## Prerequisites

- **CMake**: Version 3.10 or higher
- **CUDA Toolkit**: Compatible with NVIDIA architecture 7.5 (Turing)
- **Docker**: For containerized builds (optional)
- **Nsight Visual Studio Code Edition**: Required for CUDA debugging

## Building the Project

Follow these steps to build the project:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/albertnew2012/cpp_cuda_vscode_debug_demo.git
   cd CUDA_kernel_scatter_debug
   ```

2. **(Optional) Create a Docker container:**
   ```bash
   cd docker
   sh build_image.sh && sh create_container.sh
   ```

3. **Create a build directory:**
   ```bash
   mkdir build
   cd build
   ```

4. **Run CMake:**
   ```bash
   cmake ..
   ```

5. **Compile the project:**
   ```bash
   cmake --build .
   ```

Alternatively, you can use the provided VS Code tasks (`clean_build` or `build`) to automate the build process.

## Running the Application

After building the project, you can run the executable `kernel_debug` generated in the `build` directory:

```bash
./kernel_debug
```

## Running Tests

The project includes Google Test-based unit tests. To run the tests:

1. Navigate to the `build` directory:
   ```bash
   cd build
   ```

2. Execute the test binary:
   ```bash
   ./test_pointpillars_detection
   ```

3. Pass additional arguments (including named arguments) if needed:
   ```bash
   ./test_pointpillars_detection --input data.txt --verbose --threads 4
   ```

   Example output:
   ```
   Number of arguments received: 5
   Argument 0: ./test_pointpillars_detection
   Argument 1: --input
   Argument 2: data.txt
   Argument 3: --verbose
   Argument 4: --threads
   Named argument: input = data.txt
   Flag argument: verbose
   Named argument: threads = 4
   ```

## Debugging

The project is configured to compile in debug mode by default, which includes symbols for debugging. Use the following tools for debugging:

- **Nsight Visual Studio Code Edition**: Required for CUDA debugging.
- **cuda-gdb**: Use `"type": "cuda-gdb"` in the `launch.json` file for kernel-level debugging.

## VS Code Extensions

Install the following extensions for a better development experience:
- **CMake Tools**: For managing CMake projects.
- **Nsight Visual Studio Code Edition**: For CUDA debugging and profiling.

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the [MIT License](LICENSE).
