# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""This module contains functions related to finding compilers on the system,
and configuring Spack to use multiple compilers.
"""
import importlib
import os
import re
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import archspec.cpu

import llnl.util.filesystem as fs
import llnl.util.lang
import llnl.util.tty as tty

import spack.compiler
import spack.config
import spack.error
import spack.paths
import spack.platforms
import spack.repo
import spack.spec
import spack.version
from spack.operating_systems import windows_os
from spack.util.environment import get_path
from spack.util.naming import mod_to_class

_other_instance_vars = [
    "modules",
    "operating_system",
    "environment",
    "implicit_rpaths",
    "extra_rpaths",
]

# TODO: Caches at module level make it difficult to mock configurations in
# TODO: unit tests. It might be worth reworking their implementation.
#: cache of compilers constructed from config data, keyed by config entry id.
_compiler_cache: Dict[str, "spack.compiler.Compiler"] = {}

# TODO: generating this from the previous dict causes docs errors
package_name_to_compiler_name = {
    "llvm": "clang",
    "intel-oneapi-compilers": "oneapi",
    "llvm-amdgpu": "rocmcc",
    "intel-oneapi-compilers-classic": "intel",
    "acfl": "arm",
}


#: Tag used to identify packages providing a compiler
COMPILER_TAG = "compiler"


def _auto_compiler_spec(function):
    def converter(spec_like, *args, **kwargs):
        if not isinstance(spec_like, spack.spec.Spec):
            spec_like = spack.spec.Spec(spec_like)
        return function(spec_like, *args, **kwargs)

    return converter


def get_compiler_config(
    configuration: "spack.config.ConfigurationType", *, scope: Optional[str] = None
) -> List[Dict]:
    """Return the compiler configuration for the specified architecture."""
    compilers_yaml = configuration.get("compilers", scope=scope)
    if not compilers_yaml:
        return []
    return compilers_yaml


def get_compiler_config_from_packages(
    configuration: "spack.config.ConfigurationType",
    *,
    scope: Optional[str] = None,
    init_config: bool = False,
):
    """Return the compiler configuration from packages.yaml"""
    packages_yaml = configuration.get("packages", scope=scope)
    configs = CompilerConfigFactory.from_packages_yaml(packages_yaml)
    if configs or not init_config:
        return configs

    merged_packages_yaml = configuration.get("packages")
    configs = CompilerConfigFactory.from_packages_yaml(merged_packages_yaml)
    if configs:
        # Config is empty for this scope
        # Do not init config because there is a non-empty scope
        return configs

    find_compilers(scope=scope)
    packages_yaml = configuration.get("packages", scope=scope)
    return CompilerConfigFactory.from_packages_yaml(packages_yaml)


def compiler_config_files():
    config_files = []
    configuration = spack.config.CONFIG
    for scope in configuration.writable_scopes:
        name = scope.name

        from_packages_yaml = get_compiler_config_from_packages(configuration, scope=name)
        if from_packages_yaml:
            config_files.append(configuration.get_config_filename(name, "packages"))

        compiler_config = configuration.get("compilers", scope=name)
        if compiler_config:
            config_files.append(configuration.get_config_filename(name, "compilers"))

    return config_files


def add_compiler_to_config(compiler, scope=None) -> None:
    """Add a Compiler object to the configuration, at the required scope."""
    # FIXME (compiler as nodes): still needed to read Cray manifest
    raise NotImplementedError("'add_compiler_to_config' node implemented yet.")


def remove_compiler_from_config(compiler_spec: str, scope: Optional[str] = None) -> bool:
    """Remove compilers from the global configuration by spec.

    If scope is None, all the scopes are searched for removal.

    Arguments:
        compiler_spec: compiler to be removed
        scope: configuration scope to modify
    """
    remover = spack.compilers.CompilerRemover(spack.config.CONFIG)
    removed = remover.mark_compilers(match=compiler_spec, scope=scope)
    remover.flush()
    return bool(removed)


def all_compilers_config(
    configuration: "spack.config.ConfigurationType",
    *,
    scope: Optional[str] = None,
    init_config: bool = True,
) -> List["spack.compiler.Compiler"]:
    """Return a set of specs for all the compiler versions currently
    available to build with.  These are instances of CompilerSpec.
    """
    if os.environ.get("SPACK_EXPERIMENTAL_DEPRECATE_COMPILERS_YAML") == "1":
        from_compilers_yaml = []
    else:
        from_compilers_yaml = get_compiler_config(configuration, scope=scope)
        if from_compilers_yaml:
            init_config = False

    from_packages_yaml = get_compiler_config_from_packages(
        configuration, scope=scope, init_config=init_config
    )

    result = from_compilers_yaml + from_packages_yaml
    # Dedupe entries by the compiler they represent
    # If the entry is invalid, treat it as unique for deduplication
    key = lambda c: _compiler_from_config_entry(c["compiler"] or id(c))
    return list(llnl.util.lang.dedupe(result, key=key))


def find_compilers(
    path_hints: Optional[List[str]] = None,
    *,
    scope: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> List["spack.spec.Spec"]:
    """Searches for compiler in the paths given as argument. If any new compiler is found, the
    configuration is updated, and the list of new compiler objects is returned.

    Args:
        path_hints: list of path hints where to look for. A sensible default based on the ``PATH``
            environment variable will be used if the value is None
        scope: configuration scope to modify
        max_workers: number of processes used to search for compilers
    """
    if path_hints is None:
        path_hints = get_path("PATH")
    default_paths = fs.search_paths_for_executables(*path_hints)
    if sys.platform == "win32":
        default_paths.extend(windows_os.WindowsOs().compiler_search_paths)
    compiler_pkgs = spack.repo.PATH.packages_with_tags(COMPILER_TAG, full=True)

    detected_packages = spack.detection.by_path(
        compiler_pkgs, path_hints=default_paths, max_workers=max_workers
    )

    new_compilers = spack.detection.update_configuration(
        detected_packages, buildable=True, scope=scope
    )
    return new_compilers


def select_new_compilers(compilers, scope=None):
    """Given a list of compilers, remove those that are already defined in
    the configuration.
    """
    compilers_not_in_config = []
    for c in compilers:
        arch_spec = spack.spec.ArchSpec((None, c.operating_system, c.target))
        same_specs = compilers_for_spec(
            c.spec, arch_spec=arch_spec, scope=scope, init_config=False
        )
        if not same_specs:
            compilers_not_in_config.append(c)

    return compilers_not_in_config


def supported_compilers() -> List[str]:
    """Return a set of names of compilers supported by Spack.

    See available_compilers() to get a list of all the available
    versions of supported compilers.
    """
    # Hack to be able to call the compiler `apple-clang` while still
    # using a valid python name for the module
    return sorted(all_compiler_names())


def supported_compilers_for_host_platform() -> List[str]:
    """Return a set of compiler class objects supported by Spack
    that are also supported by the current host platform
    """
    host_plat = spack.platforms.real_host()
    return supported_compilers_for_platform(host_plat)


def supported_compilers_for_platform(platform: "spack.platforms.Platform") -> List[str]:
    """Return a set of compiler class objects supported by Spack
    that are also supported by the provided platform

    Args:
        platform (str): string representation of platform
            for which compiler compatability should be determined
    """
    return [
        name
        for name in supported_compilers()
        if class_for_compiler_name(name).is_supported_on_platform(platform)
    ]


def all_compiler_names() -> List[str]:
    def replace_apple_clang(name):
        return name if name != "apple_clang" else "apple-clang"

    return [replace_apple_clang(name) for name in all_compiler_module_names()]


@llnl.util.lang.memoized
def all_compiler_module_names() -> List[str]:
    return list(llnl.util.lang.list_modules(spack.paths.compilers_path))


@_auto_compiler_spec
def supported(compiler_spec):
    """Test if a particular compiler is supported."""
    return compiler_spec.name in supported_compilers()


@_auto_compiler_spec
def find(compiler_spec, scope=None, init_config=True):
    """Return specs of available compilers that match the supplied
    compiler spec.  Return an empty list if nothing found."""
    return [c for c in all_compilers(scope, init_config) if c.satisfies(compiler_spec)]


@_auto_compiler_spec
def find_specs_by_arch(compiler_spec, arch_spec, scope=None, init_config=True):
    """Return specs of available compilers that match the supplied
    compiler spec.  Return an empty list if nothing found."""
    return [
        c.spec
        for c in compilers_for_spec(
            compiler_spec, arch_spec=arch_spec, scope=scope, init_config=init_config
        )
    ]


def all_compilers(
    scope: Optional[str] = None, init_config: bool = True
) -> List["spack.spec.Spec"]:
    """Returns all the compilers from the current global configuration.

    Args:
        scope: configuration scope from which to extract the compilers. If None, the merged
            configuration is used.
        init_config: if True, search for compilers if none is found in configuration.
    """
    compilers = all_compilers_from(configuration=spack.config.CONFIG, scope=scope)

    if not compilers and init_config:
        find_compilers(scope=scope)
        compilers = all_compilers_from(configuration=spack.config.CONFIG, scope=scope)

    return compilers


def all_compilers_from(
    configuration: "spack.config.ConfigurationType", scope: Optional[str] = None
) -> List["spack.spec.Spec"]:
    """Returns all the compilers from the current global configuration.

    Args:
        configuration: configuration to be queried
        scope: configuration scope from which to extract the compilers. If None, the merged
            configuration is used.
    """
    compilers = []
    compiler_package_names = supported_compilers() + list(package_name_to_compiler_name.keys())

    # First, get the compilers from packages.yaml
    packages_yaml = configuration.get("packages", scope=scope)
    for name, entry in packages_yaml.items():
        if name not in compiler_package_names:
            continue

        externals_config = entry.get("externals", None)
        if not externals_config:
            continue

        compiler_specs = []
        for current_external in externals_config:
            compiler = CompilerConfigFactory._spec_from_external_config(current_external)
            if compiler:
                compiler_specs.append(compiler)

        compilers.extend(compiler_specs)

    if os.environ.get("SPACK_EXPERIMENTAL_DEPRECATE_COMPILERS_YAML") == "1":
        return compilers

    legacy_compilers = []
    for item in configuration.get("compilers", scope=scope):
        legacy_compilers.extend(_externals_from_legacy_compiler(item["compiler"]))

    if legacy_compilers:
        # FIXME (compiler as nodes): write how to update the file. Maybe an ad-hoc command
        warnings.warn(
            "Some compilers are still defined in 'compilers.yaml', which has been deprecated "
            "in v0.23. Those configuration files will be ignored from Spack v0.25.\n"
        )
        for legacy in legacy_compilers:
            if not any(c.satisfies(f"{legacy.name}@{legacy.versions}") for c in compilers):
                compilers.append(legacy)

    return compilers


class CompilerRemover:
    """Removes compiler from configuration."""

    def __init__(self, configuration: "spack.config.ConfigurationType") -> None:
        self.configuration = configuration
        self.marked_packages_yaml: List[Tuple[str, Any]] = []
        self.marked_compilers_yaml: List[Tuple[str, Any]] = []

    def mark_compilers(
        self, *, match: str, scope: Optional[str] = None
    ) -> List["spack.spec.Spec"]:
        """Marks compilers to be removed in configuration, and returns a corresponding list
        of specs.

        Args:
            match: constraint that the compiler must match to be removed.
            scope: scope where to remove the compiler. If None, all writeable scopes are checked.
        """
        self.marked_packages_yaml = []
        self.marked_compilers_yaml = []
        candidate_scopes = [scope]
        if scope is None:
            candidate_scopes = [x.name for x in self.configuration.writable_scopes]

        all_removals = self._mark_in_packages_yaml(match, candidate_scopes)
        all_removals.extend(self._mark_in_compilers_yaml(match, candidate_scopes))

        return all_removals

    def _mark_in_packages_yaml(self, match, candidate_scopes):
        compiler_package_names = supported_compilers() + list(package_name_to_compiler_name.keys())
        all_removals = []
        for current_scope in candidate_scopes:
            packages_yaml = self.configuration.get("packages", scope=current_scope)
            if not packages_yaml:
                continue

            removed_from_scope = []
            for name, entry in packages_yaml.items():
                if name not in compiler_package_names:
                    continue

                externals_config = entry.get("externals", None)
                if not externals_config:
                    continue

                def _partition_match(external_yaml):
                    s = CompilerConfigFactory._spec_from_external_config(external_yaml)
                    return not s.satisfies(match)

                to_keep, to_remove = llnl.util.lang.stable_partition(
                    externals_config, _partition_match
                )
                if not to_remove:
                    continue

                removed_from_scope.extend(to_remove)
                entry["externals"] = to_keep

            if not removed_from_scope:
                continue

            self.marked_packages_yaml.append((current_scope, packages_yaml))
            all_removals.extend(
                [CompilerConfigFactory._spec_from_external_config(x) for x in removed_from_scope]
            )
        return all_removals

    def _mark_in_compilers_yaml(self, match, candidate_scopes):
        if os.environ.get("SPACK_EXPERIMENTAL_DEPRECATE_COMPILERS_YAML") == "1":
            return []

        all_removals = []
        for current_scope in candidate_scopes:
            compilers_yaml = self.configuration.get("compilers", scope=current_scope)
            if not compilers_yaml:
                continue

            def _partition_match(entry):
                external_specs = _externals_from_legacy_compiler(entry["compiler"])
                return not any(x.satisfies(match) for x in external_specs)

            to_keep, to_remove = llnl.util.lang.stable_partition(compilers_yaml, _partition_match)
            if not to_remove:
                continue

            compilers_yaml[:] = to_keep
            self.marked_compilers_yaml.append((current_scope, compilers_yaml))
            for entry in to_remove:
                all_removals.extend(_externals_from_legacy_compiler(entry["compiler"]))

        return all_removals

    def flush(self):
        """Removes from configuration the specs that have been marked by the previous call
        of ``remove_compilers``.
        """
        for scope, packages_yaml in self.marked_packages_yaml:
            self.configuration.set("packages", packages_yaml, scope=scope)

        for scope, compilers_yaml in self.marked_compilers_yaml:
            self.configuration.set("compilers", compilers_yaml, scope=scope)


def _externals_from_legacy_compiler(compiler_dict: Dict[str, Any]) -> List["spack.spec.Spec"]:
    """Returns a list of external specs, corresponding to a compiler entry from compilers.yaml."""
    from spack.detection.path import ExecutablesFinder

    result = []
    candidate_paths = [x for x in compiler_dict["paths"].values() if x is not None]
    finder = ExecutablesFinder()
    for pkg_name in spack.repo.PATH.packages_with_tags("compiler"):
        pkg_cls = spack.repo.PATH.get_pkg_class(pkg_name)
        pattern = re.compile(r"|".join(finder.search_patterns(pkg=pkg_cls)))
        filtered_paths = [x for x in candidate_paths if pattern.search(os.path.basename(x))]
        detected = finder.detect_specs(pkg=pkg_cls, paths=filtered_paths)
        if detected:
            for item in detected:
                spec, prefix = item.spec, item.prefix
                spec.external_path = prefix
                result.append(spec)

    return result


@_auto_compiler_spec
def compilers_for_spec(compiler_spec, *, arch_spec=None, scope=None, init_config=True):
    """This gets all compilers that satisfy the supplied CompilerSpec.
    Returns an empty list if none are found.
    """

    config = all_compilers_config(spack.config.CONFIG, scope=scope, init_config=init_config)
    matches = set(find(compiler_spec, scope, init_config))
    compilers = []
    for cspec in matches:
        compilers.extend(get_compilers(config, cspec, arch_spec))
    return compilers


def compilers_for_arch(arch_spec, scope=None):
    # FIXME (compiler as nodes): this needs a better implementation
    compilers = all_compilers_from(spack.config.CONFIG, scope=scope)
    result = []
    for candidate in compilers:
        _, operating_system, target = name_os_target(candidate)
        if not operating_system == str(arch_spec.os) or not target == str(
            arch_spec.target.microarchitecture.family
        ):
            continue
        result.append(candidate)
    return result


class CacheReference:
    """This acts as a hashable reference to any object (regardless of whether
    the object itself is hashable) and also prevents the object from being
    garbage-collected (so if two CacheReference objects are equal, they
    will refer to the same object, since it will not have been gc'ed since
    the creation of the first CacheReference).
    """

    def __init__(self, val):
        self.val = val
        self.id = id(val)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, CacheReference) and self.id == other.id


def compiler_from_dict(items):
    cspec = spack.spec.parse_with_version_concrete(items["spec"])
    os = items.get("operating_system", None)
    target = items.get("target", None)

    if not (
        "paths" in items and all(n in items["paths"] for n in spack.compiler.PATH_INSTANCE_VARS)
    ):
        raise InvalidCompilerConfigurationError(cspec)

    cls = class_for_compiler_name(cspec.name)

    compiler_paths = []
    for c in spack.compiler.PATH_INSTANCE_VARS:
        compiler_path = items["paths"][c]
        if compiler_path != "None":
            compiler_paths.append(compiler_path)
        else:
            compiler_paths.append(None)

    mods = items.get("modules")
    if mods == "None":
        mods = []

    alias = items.get("alias", None)
    compiler_flags = items.get("flags", {})
    environment = items.get("environment", {})
    extra_rpaths = items.get("extra_rpaths", [])
    implicit_rpaths = items.get("implicit_rpaths", None)

    # Starting with c22a145, 'implicit_rpaths' was a list. Now it is a
    # boolean which can be set by the user to disable all automatic
    # RPATH insertion of compiler libraries
    if implicit_rpaths is not None and not isinstance(implicit_rpaths, bool):
        implicit_rpaths = None

    return cls(
        cspec,
        os,
        target,
        compiler_paths,
        mods,
        alias,
        environment,
        extra_rpaths,
        enable_implicit_rpaths=implicit_rpaths,
        **compiler_flags,
    )


def _compiler_from_config_entry(items):
    """Note this is intended for internal use only. To avoid re-parsing
    the same config dictionary this keeps track of its location in
    memory. If you provide the same dictionary twice it will return
    the same Compiler object (regardless of whether the dictionary
    entries have changed).
    """
    config_id = CacheReference(items)
    compiler = _compiler_cache.get(config_id, None)

    if compiler is None:
        try:
            compiler = compiler_from_dict(items)
        except UnknownCompilerError as e:
            warnings.warn(e.message)
        _compiler_cache[config_id] = compiler

    return compiler


def get_compilers(config, cspec=None, arch_spec=None):
    compilers = []

    for items in config:
        items = items["compiler"]

        # We might use equality here.
        if cspec and not spack.spec.parse_with_version_concrete(
            items["spec"], compiler=True
        ).satisfies(cspec):
            continue

        # If an arch spec is given, confirm that this compiler
        # is for the given operating system
        os = items.get("operating_system", None)
        if arch_spec and os != arch_spec.os:
            continue

        # If an arch spec is given, confirm that this compiler
        # is for the given target. If the target is 'any', match
        # any given arch spec. If the compiler has no assigned
        # target this is an old compiler config file, skip this logic.
        target = items.get("target", None)

        try:
            current_target = archspec.cpu.TARGETS[str(arch_spec.target)]
            family = str(current_target.family)
        except KeyError:
            # TODO: Check if this exception handling makes sense, or if we
            # TODO: need to change / refactor tests
            family = arch_spec.target
        except AttributeError:
            assert arch_spec is None

        if arch_spec and target and (target != family and target != "any"):
            # If the family of the target is the family we are seeking,
            # there's an error in the underlying configuration
            if archspec.cpu.TARGETS[target].family == family:
                msg = (
                    'the "target" field in compilers.yaml accepts only '
                    'target families [replace "{0}" with "{1}"'
                    ' in "{2}" specification]'
                )
                msg = msg.format(str(target), family, items.get("spec", "??"))
                raise ValueError(msg)
            continue

        compiler = _compiler_from_config_entry(items)
        if compiler:
            compilers.append(compiler)

    return compilers


@_auto_compiler_spec
def compiler_for_spec(compiler_spec, arch_spec):
    """Get the compiler that satisfies compiler_spec.  compiler_spec must
    be concrete."""
    assert compiler_spec.concrete
    assert arch_spec.concrete

    compilers = compilers_for_spec(compiler_spec, arch_spec=arch_spec)
    if len(compilers) < 1:
        raise NoCompilerForSpecError(compiler_spec, arch_spec.os)
    if len(compilers) > 1:
        msg = "Multiple definitions of compiler %s " % compiler_spec
        msg += "for architecture %s:\n %s" % (arch_spec, compilers)
        tty.debug(msg)
    return compilers[0]


@_auto_compiler_spec
def get_compiler_duplicates(compiler_spec, arch_spec):
    config = spack.config.CONFIG

    scope_to_compilers = {}
    for scope in config.scopes:
        compilers = compilers_for_spec(compiler_spec, arch_spec=arch_spec, scope=scope)
        if compilers:
            scope_to_compilers[scope] = compilers

    cfg_file_to_duplicates = {}
    for scope, compilers in scope_to_compilers.items():
        config_file = config.get_config_filename(scope, "compilers")
        cfg_file_to_duplicates[config_file] = compilers

    return cfg_file_to_duplicates


@llnl.util.lang.memoized
def class_for_compiler_name(compiler_name):
    """Given a compiler module name, get the corresponding Compiler class."""
    if not supported(compiler_name):
        raise UnknownCompilerError(compiler_name)

    # Hack to be able to call the compiler `apple-clang` while still
    # using a valid python name for the module
    submodule_name = compiler_name
    if compiler_name == "apple-clang":
        submodule_name = compiler_name.replace("-", "_")

    module_name = ".".join(["spack", "compilers", submodule_name])
    module_obj = importlib.import_module(module_name)
    cls = getattr(module_obj, mod_to_class(compiler_name))

    # make a note of the name in the module so we can get to it easily.
    cls.name = compiler_name

    return cls


def all_compiler_types():
    return [class_for_compiler_name(c) for c in supported_compilers()]


def is_mixed_toolchain(compiler):
    """Returns True if the current compiler is a mixed toolchain,
    False otherwise.

    Args:
        compiler (spack.compiler.Compiler): a valid compiler object
    """
    cc = os.path.basename(compiler.cc or "")
    cxx = os.path.basename(compiler.cxx or "")
    f77 = os.path.basename(compiler.f77 or "")
    fc = os.path.basename(compiler.fc or "")

    toolchains = set()
    for compiler_cls in all_compiler_types():
        # Inspect all the compiler toolchain we know. If a compiler is the
        # only compiler supported there it belongs to that toolchain.
        def name_matches(name, name_list):
            # This is such that 'gcc' matches variations
            # like 'ggc-9' etc that are found in distros
            name, _, _ = name.partition("-")
            return len(name_list) == 1 and name and name in name_list

        if any(
            [
                name_matches(cc, compiler_cls.cc_names),
                name_matches(cxx, compiler_cls.cxx_names),
                name_matches(f77, compiler_cls.f77_names),
                name_matches(fc, compiler_cls.fc_names),
            ]
        ):
            tty.debug("[TOOLCHAIN] MATCH {0}".format(compiler_cls.__name__))
            toolchains.add(compiler_cls.__name__)

    if len(toolchains) > 1:
        if (
            toolchains == set(["Clang", "AppleClang", "Aocc"])
            # Msvc toolchain uses Intel ifx
            or toolchains == set(["Msvc", "Dpcpp", "Oneapi"])
        ):
            return False
        tty.debug("[TOOLCHAINS] {0}".format(toolchains))
        return True

    return False


_EXTRA_ATTRIBUTES_KEY = "extra_attributes"
_COMPILERS_KEY = "compilers"
_C_KEY = "c"
_CXX_KEY, _FORTRAN_KEY = "cxx", "fortran"


def name_os_target(spec: "spack.spec.Spec") -> Tuple[str, str, str]:
    return (spec.name,) + CompilerConfigFactory._extract_os_and_target(spec)


class CompilerConfigFactory:
    """Class aggregating all ways of constructing a list of compiler config entries."""

    @staticmethod
    def from_specs(specs: List["spack.spec.Spec"]) -> List[dict]:
        result = []
        compiler_package_names = supported_compilers() + list(package_name_to_compiler_name.keys())
        for s in specs:
            if s.name not in compiler_package_names:
                continue

            candidate = CompilerConfigFactory.from_external_spec(s)
            if candidate is None:
                continue

            result.append(candidate)
        return result

    @staticmethod
    def from_packages_yaml(packages_yaml) -> List["spack.spec.Spec"]:
        compiler_specs = []
        compiler_package_names = supported_compilers() + list(package_name_to_compiler_name.keys())
        for name, entry in packages_yaml.items():
            if name not in compiler_package_names:
                continue

            externals_config = entry.get("externals", None)
            if not externals_config:
                continue

            current_specs = []
            for current_external in externals_config:
                compiler = CompilerConfigFactory._spec_from_external_config(current_external)
                if compiler:
                    current_specs.append(compiler)
            compiler_specs.extend(current_specs)

        return compiler_specs

    @staticmethod
    def _spec_from_external_config(config) -> Optional["spack.spec.Spec"]:
        # Allow `@x.y.z` instead of `@=x.y.z`
        err_header = f"The external spec '{config['spec']}' cannot be used as a compiler"
        # If extra_attributes is not there I might not want to use this entry as a compiler,
        # therefore just leave a debug message, but don't be loud with a warning.
        if _EXTRA_ATTRIBUTES_KEY not in config:
            tty.debug(f"[{__file__}] {err_header}: missing the '{_EXTRA_ATTRIBUTES_KEY}' key")
            return None
        extra_attributes = config[_EXTRA_ATTRIBUTES_KEY]
        result = spack.spec.Spec(
            str(spack.spec.parse_with_version_concrete(config["spec"])),
            external_path=config.get("prefix"),
            external_modules=config.get("modules"),
        )
        result.extra_attributes = extra_attributes
        return result

    @staticmethod
    def from_external_spec(spec: "spack.spec.Spec") -> Optional[dict]:
        spec = spack.spec.parse_with_version_concrete(spec)
        extra_attributes = getattr(spec, _EXTRA_ATTRIBUTES_KEY, None)
        if extra_attributes is None:
            return None

        paths = CompilerConfigFactory._extract_compiler_paths(spec)
        if paths is None:
            return None

        compiler_spec = spack.spec.CompilerSpec(
            package_name_to_compiler_name.get(spec.name, spec.name), spec.version
        )

        operating_system, target = CompilerConfigFactory._extract_os_and_target(spec)

        compiler_entry = {
            "compiler": {
                "spec": str(compiler_spec),
                "paths": paths,
                "flags": extra_attributes.get("flags", {}),
                "operating_system": str(operating_system),
                "target": str(target.family),
                "modules": getattr(spec, "external_modules", []),
                "environment": extra_attributes.get("environment", {}),
                "extra_rpaths": extra_attributes.get("extra_rpaths", []),
                "implicit_rpaths": extra_attributes.get("implicit_rpaths", None),
            }
        }
        return compiler_entry

    @staticmethod
    def _extract_compiler_paths(spec: "spack.spec.Spec") -> Optional[Dict[str, str]]:
        err_header = f"The external spec '{spec}' cannot be used as a compiler"
        extra_attributes = spec.extra_attributes
        # If I have 'extra_attributes' warn if 'compilers' is missing,
        # or we don't have a C compiler
        if _COMPILERS_KEY not in extra_attributes:
            warnings.warn(
                f"{err_header}: missing the '{_COMPILERS_KEY}' key under '{_EXTRA_ATTRIBUTES_KEY}'"
            )
            return None
        attribute_compilers = extra_attributes[_COMPILERS_KEY]

        if _C_KEY not in attribute_compilers:
            warnings.warn(
                f"{err_header}: missing the C compiler path under "
                f"'{_EXTRA_ATTRIBUTES_KEY}:{_COMPILERS_KEY}'"
            )
            return None
        c_compiler = attribute_compilers[_C_KEY]

        # C++ and Fortran compilers are not mandatory, so let's just leave a debug trace
        if _CXX_KEY not in attribute_compilers:
            tty.debug(f"[{__file__}] The external spec {spec} does not have a C++ compiler")

        if _FORTRAN_KEY not in attribute_compilers:
            tty.debug(f"[{__file__}] The external spec {spec} does not have a Fortran compiler")

        # compilers format has cc/fc/f77, externals format has "c/fortran"
        return {
            "cc": c_compiler,
            "cxx": attribute_compilers.get(_CXX_KEY, None),
            "fc": attribute_compilers.get(_FORTRAN_KEY, None),
            "f77": attribute_compilers.get(_FORTRAN_KEY, None),
        }

    @staticmethod
    def _extract_os_and_target(spec: "spack.spec.Spec"):
        if not spec.architecture:
            host_platform = spack.platforms.host()
            operating_system = host_platform.operating_system("default_os")
            target = host_platform.target("default_target").microarchitecture
        else:
            target = spec.architecture.target
            if not target:
                target = spack.platforms.host().target("default_target")
            target = target.microarchitecture

            operating_system = spec.os
            if not operating_system:
                host_platform = spack.platforms.host()
                operating_system = host_platform.operating_system("default_os")
        return str(operating_system), str(target.family)


class InvalidCompilerConfigurationError(spack.error.SpackError):
    def __init__(self, compiler_spec):
        super().__init__(
            f'Invalid configuration for [compiler "{compiler_spec}"]: ',
            f"Compiler configuration must contain entries for "
            f"all compilers: {spack.compiler.PATH_INSTANCE_VARS}",
        )


class UnknownCompilerError(spack.error.SpackError):
    def __init__(self, compiler_name):
        super().__init__("Spack doesn't support the requested compiler: {0}".format(compiler_name))


class NoCompilerForSpecError(spack.error.SpackError):
    def __init__(self, compiler_spec, target):
        super().__init__(
            "No compilers for operating system %s satisfy spec %s" % (target, compiler_spec)
        )
