from os.path import splitext
import sysconfig


def get_soname():
    soname = sysconfig.get_config_vars("INSTSONAME")
    name, ext = splitext(soname[0])
    if name.startswith("lib"):
        return name[3:]
    return name


def get_sopath():
    ldir = sysconfig.get_config_vars("LIBDIR")
    return ldir[0]


if __name__ == "__main__":

    print(f"cargo:rustc-link-search={get_sopath()}")
    print(f"cargo:rustc-link-lib={get_soname()}")
