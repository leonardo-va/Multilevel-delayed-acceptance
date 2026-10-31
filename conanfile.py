from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps


class MultiLevelSampler(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    # generators = "CMakeToolchain", "CMakeDeps"

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["BUILD_TESTS"] = True
        tc.cache_variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def requirements(self):
        self.requires("boost/1.91.0")
        self.requires("spdlog/1.17.0")
        self.requires("indicators/2.3")
        self.requires("magic_enum/0.9.7")
        self.requires("nlohmann_json/3.12.0")
        self.requires("eigen/5.0.1")

    # def build_requirements(self):
    #     self.tool_requires("cmake/4.3.1")

