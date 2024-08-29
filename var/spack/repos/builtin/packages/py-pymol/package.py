# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class PyPymol(PythonPackage):
    """PyMOL is a Python-enhanced molecular graphics tool. It excels at 3D
    visualization of proteins, small molecules, density, surfaces, and
    trajectories. It also includes molecular editing, ray tracing, and
    movies. Open Source PyMOL is free to everyone!"""

    homepage = "https://pymol.org"
    url = "https://github.com/schrodinger/pymol-open-source/archive/refs/tags/v3.0.0.tar.gz"

    version("3.0.0", sha256="45e800a02680cec62dff7a0f92283f4be7978c13a02934a43ac6bb01f67622cf")
    version("2.5.0", sha256="aa828bf5719bd9a14510118a93182a6e0cadc03a574ba1e327e1e9780a0e80b3")
    version("2.4.0", sha256="5ede4ce2e8f53713c5ee64f5905b2d29bf01e4391da7e536ce8909d6b9116581")
    version("2.3.0", sha256="62aa21fafd1db805c876f89466e47513809f8198395e1f00a5f5cc40d6f40ed0")

    depends_on("c", type="build")  # generated
    depends_on("cxx", type="build")  # generated

    # the default python build has an extended version of tkniter as it's default.
    # requesting python with the tkniter vairant seems to be what is causing the cyclic dependency
    # I simply removed the dependency on the variant and it built and ran properly (at least with 
    # my limited testing of the 3.0.0 version)
    depends_on("python@3.9:", type=("build", "link", "run"),)
    # in newer pip versions --install-option does not exist
    #depends_on("py-pip@:23.0", type="build")
    depends_on("py-setuptools@69.2.0:", type="build")
    depends_on("gl")
    depends_on("glew")
    depends_on("libpng")
    depends_on("freetype")
    depends_on("glm")
    depends_on("libmmtf-cpp")
    depends_on("msgpack-c@2.1.5:")
    depends_on("netcdf-cxx4")
    depends_on("libxml2")
    depends_on("py-pmw-patched", type=("build", "run"))
    depends_on("py-pyqt5", type=("build", "run"))
    depends_on("py-pmw", type=("build", "run"))
    depends_on("libmmtf-cpp", type=("build", "run", "link"))
    depends_on("msgpack-c", type=("build", "run"))
    depends_on("libpng", type=("build", "run"))
    depends_on("py-numpy@1.26.4:2", type=("build", "link", "run"))
    depends_on("py-msgpack", type=("build", "run"))

    def install_options(self, spec, prefix):
        return ["--no-launcher"]

    def install(self, spec, prefix):
        # Note: pymol monkeypatches distutils which breaks pip install, use deprecated
        # `python setup.py install` and distutils instead of `pip install` and
        # setuptools. See: https://github.com/schrodinger/pymol-open-source/issues/217
        python("setup.py", "install", "--prefix=" + prefix, *self.install_options(spec, prefix))

    @run_after("install")
    def install_launcher(self):
        binpath = self.prefix.bin
        mkdirp(self.prefix.bin)
        fname = join_path(binpath, "pymol")
        script = join_path(python_platlib, "pymol", "__init__.py")

        shebang = "#!/bin/sh\n"
        fdata = f'exec {python.path} {script} "$@"'
        with open(fname, "w") as new:
            new.write(shebang + fdata)
        set_executable(fname)
