# Multilevel Delayed Acceptance Sampler

## Requirements

- Python >= 3.10
- Conan >= 2.0
- CMake >= 4.3.1
- C++23-compatible compiler

## Setup

### 1. Create Python environment

    python3 -m venv .venv
    source .venv/bin/activate

### 2. Install Conan

    pip install --upgrade pip
    pip install "conan>=2.0"

### 3. Configure Conan

    conan profile detect --force

### 4. Install dependencies

Run from the project root directory:

    conan install . --output-folder=build --build=missing -s build_type=Release

### 5. Configure and build

    cmake -S . -B build \
        -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release

    cmake --build build --parallel

### 6. Run tests

    ctest --test-dir build --output-on-failure

## Notes

- Dependencies are managed through `conanfile.py`.
- Conan generates the CMake toolchain and dependency configuration files.
- Missing dependency binaries are automatically built from source.
- CMake must be installed separately; it is not currently managed by Conan.
- Tests are enabled by default (`BUILD_TESTS=ON`).
