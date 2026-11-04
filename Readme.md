# Multilevel Delayed Acceptance Sampler

Dependencies are managed with Conan 2, which requires Python to be installed.\
Build is done with CMake and a C++23-compatible compiler.

## Requirements

- Python \>= 3.10
- Conan \>= 2.0
- CMake \>= 4.3.1
- C++23-compatible compiler

## Setup

### 1\. Install Conan

Install Conan using :


```bash
python -m pip install "conan>=2.0"
```

Verify the installation:


```bash
conan --version
```

### 2\. Configure the Conan profile

If this is your first time using Conan on this system, detect the available compiler and generate a default profile:


```bash
conan profile detect
```

This detects the installed compiler, its version, the system architecture and operating system. The generated profile is stored in Conan's configuration directory.

You can inspect the detected profile using:


```bash
conan profile show
```

### 3\. Install dependencies

From the project working directory containing `conanfile.py`, run:


```bash
conan install . -of ./build --build=missing
```

This resolves and installs the dependencies specified in `conanfile.py`. Existing compatible binary packages are downloaded, while dependencies without an available binary package are built from source.

### 4\. Configure and build

Configure the project using the Conan-generated CMake preset:


```bash
cmake --preset conan-release
```

Build the project:


```bash
cmake --build --preset conan-release
```

### 5. Run the example program

After successfully building the project, the example executable can be found in the `build/bin` directory.

Run the program from the project root directory using:

    ./build/bin/thesis_example -i <path-to-input-file> -o <path-to-output-folder>

The program accepts the following command-line arguments:

| Argument | Description | Required |
|----------|-------------|----------|
| `-i, --input` | Path to the input data file (must exist). | Yes |
| `-o, --output` | Path to the output directory. | Yes |

For example:

    ./build/bin/thesis_example -i ./dataJM.csv -o ./out


**Note:** Both arguments must be provided when running the program. The input file must already exist. The output path specifies the directory where the generated results will be stored.

