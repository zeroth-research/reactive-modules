"""The project on disk, written by `create_project`: its directories, the
static Lean library copied in, the lakefile and toolchain, the module's
encodings (`ENCODINGS`) and the certificate's files.
`generate_standalone_cert_lean` is the one-file form `--cert-file` writes.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from zrth import Module

from ..codegen.context import LeanContext
from ..codegen.encodings import ModuleToLean4
from .names import (
    DATA, LINTERS_OFF, PROOF, SYSTEM_LAYOUT, Layout, cert_file,
)
from .templates import STATIC_DIR, render

from ..proof.render import (
    LeanCertificate,
    generate_certificate_lean,
    generate_data_lean,
)
from .lakefile import LEAN_TOOLCHAIN, LakefileParts, generate_lakefile
from .main_lean import generate_main_lean

if TYPE_CHECKING:
    from ..inputs import SourceText


# Static template files (copied as-is into the project)
TEMPLATE_DIR = STATIC_DIR
CORE_FILES = ["Basic.lean", "Box.lean", "LTL.lean", "Mat.lean"]
LEAN_AI_FILES = ["LeanAI.lean", "LeanAI"]


# ══════════════════════════════════════════════════════════════════════════
# The emitted encodings
#
# One row per Lean file `ModuleToLean4` prints, and one writer for all of
# them. Both routes that emit encodings -- the project (`create_project`)
# and the standalone `--cert-file` -- walk this same table, so the header,
# the imports and the producer agree across the two by construction rather
# than by inspection. They differ only in how a row becomes a path and a
# Lean module name, which is what `Layout` supplies.
#
# Row order is emission order only: each producer prints its definitions
# and the theorems about them in one call, so no row reads state another
# left behind. Rows are coupled by Lean names -- `ScalarRel` rewrites by
# `Scalar`'s `init_scalar_eq`, every row quotes the base's `init`/`update`
# -- and `Encoding.needs`, which becomes each file's imports, records that.
# ══════════════════════════════════════════════════════════════════════════




@dataclass(frozen=True)
class Encoding:
    """One Lean file: what names it, what it says, what it needs, who prints it."""

    suffix: str                 # appended to the layout's base; "" is the base
    title: str                  # header comment, "<title> of reactive module `X`"
    core: str                   # the Core module the encoding is written against
    needs: tuple[str, ...]      # suffixes of the sibling encodings it imports
    producer: str               # the `ModuleToLean4` method that prints it
    cert_file: bool = True      # also written by the `--cert-file` route


ENCODINGS: tuple[Encoding, ...] = (
    # The functional encoding is the base every other row imports. The
    # `--cert-file` route gets its own copy inlined into the standalone
    # certificate by `generate_standalone_cert_lean`, so it is not written
    # again there; `Circ` has no consumer on that route at all.
    Encoding("", "Functional encoding", "Core.Box", (), "to_lean_functional",
             cert_file=False),
    Encoding("Circ", "Circuit encoding", "Core.Box", ("",), "to_lean_circ",
             cert_file=False),
    Encoding("Rel", "Matrix-domain relational encoding", "Core.Basic", ("",),
             "to_lean_mat_rel"),
    Encoding("Scalar", "Scalar encoding", "Core.Basic", ("",), "to_lean_scalar"),
    Encoding("ScalarRel", "Scalar-relational encoding", "Core.Basic", ("Scalar",),
             "to_lean_rel"),
)


def write_encoding(
    enc: Encoding, m2l: "ModuleToLean4", layout: Layout, module_name: str
) -> Path:
    """Print one encoding and write it where `layout` says it goes."""
    imports = [enc.core] + [layout.lean_module(s) for s in enc.needs]
    body = getattr(m2l, enc.producer)()
    path = layout.path(enc.suffix)
    path.write_text(
        f"/- {enc.title} of reactive module `{module_name}` -/\n"
        + "".join(f"import {i}\n" for i in imports)
        + f"\n{LINTERS_OFF}\n\n{body}\n"
    )
    return path


def generate_root() -> str:
    """Root module file that imports the `System.*` encodings.

    The import list is the encoding table, so a row added there reaches the
    root without a second edit.
    """
    return render(
        "project/Root.lean.j2",
        imports=[SYSTEM_LAYOUT.lean_module(e.suffix) for e in ENCODINGS],
    )


def generate_zeroth_hammer_lean() -> str:
    """Generate a standalone ZerothHammer.lean with only the zeroth_hammer tactic."""
    return render("project/ZerothHammer.lean.j2")


def write_data_lean(
    project_dir: Path,
    project_name: str,
    module: Module,
    lean_cert: LeanCertificate | None,
    ctx: LeanContext | None = None,
) -> Path:
    """Write/overwrite Certificate/Data.lean with init_pre, update_pre, inv, P,
    ranking.

    Pass a pre-built ``ctx`` to avoid rebuilding LeanContext (e.g. when called
    from ``create_project`` which already has one).

    Beside ``Certificate.lean`` rather than under ``System/``: both files are
    about the certificate and both are rewritten whenever the predicates
    change, while ``System/`` holds the module, which does not. The directory
    is made here because this can be called on its own, after inference, and
    is not guaranteed to run after ``write_certificate_lean``.
    """
    if ctx is None:
        ctx = LeanContext(module)

    data_file = project_dir / cert_file(DATA)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text(generate_data_lean(ctx, lean_cert))
    print(f"Wrote {data_file}")
    return data_file


def write_certificate_lean(
    project_dir: Path,
    project_name: str,
    module: Module,
    lean_cert: LeanCertificate | None = None,
    ctx: LeanContext | None = None,
) -> Path:
    """Write/overwrite Certificate.lean (stable proof structure, imports Data).

    Pass a pre-built ``ctx`` to avoid rebuilding LeanContext (e.g. when called
    from ``create_project`` which already has one).

    The *definitions* live in ``Certificate/Data.lean`` beside it, but the
    proof tactics are generated from the shape of the predicates (see
    ``zrth.lean.proof.tactics``), so this file has to be rewritten whenever they
    change — after ``--infer``, for instance.
    """
    if ctx is None:
        ctx = LeanContext(module)

    path = project_dir / cert_file(PROOF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_certificate_lean(ctx, lean_cert))
    print(f"Wrote {path}")
    return path


def create_project(
    output_dir: Path,
    module: Module,
    project_name: str = "Certificate",
    template_dir: Path = TEMPLATE_DIR,
    executable: bool = False,
    lean_cert: LeanCertificate | None = None,
    source: "SourceText | None" = None,
    lakefile: LakefileParts | None = None,
    remove: Iterable[str] = (),
    ctx: "LeanContext | None" = None,
) -> Path:
    """
    Create a full Lean4 project.

    Args:
        `output_dir`:      Where to create the project folder.
        `diagram_sources`: Dict of {filename_stem: lean_source_code}
                           e.g. {"MixedDiagram": "import ...\\ndef mixedDiagram ..."}
        `template_dir`:    Optional path to directory containing template .lean files.
        `project_name`:    Optional name of the Lean package / library.
        `executable`:      If True, generate Main.lean and add [[lean_exe]] to lakefile.
        `lean_cert`:       The certificate's predicates, printed; placeholders
                           where it is `None`.
        `source`:          The input the module was read from, written to
                           `dbg/system<suffix>` as a debug artifact.
        `lakefile`:        What the inference route adds to the lakefile,
                           from its row's `lakefile` hook. Declared as data
                           rather than passed per route: the lakefile is
                           written here, before any route runs, so a route
                           cannot amend it afterwards.
        `remove`:          Project-relative files an earlier run in the same
                           `-o` may have left where a lean_lib globs; they
                           are deleted before anything is written there.
        `ctx`:             A `LeanContext` this run already built, so the
                           discovery pass runs once rather than once here
                           and again for every file rewritten after
                           inference.
    """
    project_dir = output_dir / project_name
    src_dir = project_dir / "System"
    dbg_dir = project_dir / "dbg"
    module_name = project_name

    # Create directory structure
    src_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created project directory: `{project_dir}`")

    # Debug artifacts: module representation and the input it was read from
    dbg_dir.mkdir(parents=True, exist_ok=True)
    (dbg_dir / "system.txt").write_text(str(module))
    print("Wrote system.txt")
    if source is not None:
        copy = f"system{source.suffix}"
        (dbg_dir / copy).write_text(source.text)
        print(f"Wrote {copy}")

    # Render and write project-level files from templates
    lakefile_path = project_dir / "lakefile.toml"
    lakefile_path.write_text(
        generate_lakefile(
            project_name, executable=executable,
            parts=lakefile or LakefileParts(),
        )
    )
    print(f"Wrote {lakefile_path}")

    toolchain = project_dir / "lean-toolchain"
    toolchain.write_text(
        render("project/lean-toolchain.j2", lean_toolchain=LEAN_TOOLCHAIN)
    )
    print(f"Wrote {toolchain}")

    hammer_file = project_dir / "ZerothHammer.lean"
    hammer_file.write_text(generate_zeroth_hammer_lean())
    print(f"Wrote {hammer_file}")

    (project_dir / "Certificate.lean").write_text(render("project/Certificate.lean.j2"))

    # The `Certificate` lean_lib globs its submodules, so every file in
    # `Certificate/` is built -- including one this run did not write. A file
    # an inference route writes is written *after* this, so anything already
    # there is a copy an earlier run left in the same `-o`: it is about a
    # different module, and lake would build it against this run's `System/`.
    # The caller names those files in `remove`.
    for rel in remove:
        (project_dir / rel).unlink(missing_ok=True)

    # Copy static files (Core/, LeanAI/)
    core_dir = project_dir / "Core"
    core_dir.mkdir(parents=True, exist_ok=True)
    for tmpl_name in CORE_FILES:
        src_path = template_dir / "Core" / tmpl_name
        dst_path = core_dir / tmpl_name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"Copied {tmpl_name} -> Core/")
        else:
            raise RuntimeError(f"Template file `{tmpl_name}` not found at {src_path}")

    for tmpl_name in LEAN_AI_FILES:
        src_path = template_dir / tmpl_name
        dst_path = project_dir / tmpl_name
        if src_path.exists():
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Copied template directory {tmpl_name} -> /")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"Copied template file {tmpl_name} -> /")
        else:
            raise RuntimeError(
                f"Template file/dir `{tmpl_name}` not found at {src_path}"
            )

    # ----------------------------------------------------------
    # Generate reactive module (init and update)
    # ----------------------------------------------------------
    if ctx is None:
        ctx = LeanContext(module)
    m2l = ModuleToLean4(ctx)

    root_lean = project_dir / "System.lean"
    root_lean.write_text(generate_root())
    print(f"Wrote root module {root_lean}")

    # Every encoding the project carries, in table order.
    layout = replace(SYSTEM_LAYOUT, directory=src_dir)
    for enc in ENCODINGS:
        print(f"++ Generated {write_encoding(enc, m2l, layout, module_name)} ++")

    # -- certificate data (init_pre, inv, P, ranking — placeholders if no lean_cert) --
    write_data_lean(project_dir, project_name, module, lean_cert, ctx=ctx)

    # ----------------------------------------------------------
    # Generate Main.lean for executable
    # ----------------------------------------------------------
    if executable:
        main_lean = project_dir / "Main.lean"
        main_lean.write_text(generate_main_lean(project_name, module, module_name))
        print(f"Wrote {main_lean}")

    # ----------------------------------------------------------
    # Always write Certificate.lean (stable proof structure, imports XXXData)
    # ----------------------------------------------------------
    write_certificate_lean(project_dir, project_name, module, lean_cert, ctx=ctx)

    print()
    print(f"DONE: Project created at: {project_dir}")
    return project_dir


def generate_standalone_cert_lean(
    module: Module,
    lean_cert: LeanCertificate | None = None,
) -> str:
    """Generate a self-contained certificate Lean file with init/update inlined.

    Unlike :func:`write_certificate_lean`, the functional module code
    (init, update) is inlined directly rather than imported.  ZerothHammer
    is always imported from the project-level ``ZerothHammer.lean``.

    Suitable for placing a single .lean file inside an existing lake project
    such as ``tests/lean/Certs/``.
    """
    ctx = LeanContext(module)
    m2l = ModuleToLean4(ctx)
    module_code = m2l.to_lean_functional()
    return generate_certificate_lean(ctx, lean_cert, module_inline=module_code)
