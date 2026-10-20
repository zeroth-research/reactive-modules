"""What a run writes: the project (`generate`), a standalone certificate
(`standalone`, `--cert-file`), the hammer tactic alone (`hammer_only`,
`--hammer-file`), and the `lake build` `--build-cert` asks for
(`build_cert`).
"""

from __future__ import annotations

from pathlib import Path

from ..cli import Settings
from ..codegen.context import LeanContext
from ..codegen.encodings import ModuleToLean4
from ..core.errors import Refused
from .. import inputs
from ..infer import owned_files
from ..project.build import LakeBuildError, build_certificate
from ..project.scaffold import (
    ENCODINGS,
    Layout,
    create_project,
    generate_zeroth_hammer_lean,
    write_encoding,
)
from ..proof.render import (
    LeanCertificate,
    certificate_files,
    generate_standalone_cert_lean,
)


def build_cert(project_dir: Path, lean_cert: "LeanCertificate | None") -> None:
    """`--build-cert`, once the certificate is written.

    `lean_cert` is what the project's certificate was generated from, or
    `None` when the certificate came from elsewhere (`--infer fbk-proveit`
    installs ic3ia's). The predicates are re-checked here rather than only
    at parse time because a route can return without one: an LLM that
    answers nothing usable leaves `inv` or `ranking` unset, and building
    that would report a proof of `sorry`.
    """
    if lean_cert is not None:
        defs = lean_cert.defs
        # A safety certificate has no ranking function to be missing.
        required = [("property", defs.prp), ("invariant", defs.inv)]
        if not defs.is_safety:
            required.append(("ranking", defs.ranking))
        missing = [name for name, value in required if not value]
        if missing:
            raise SystemExit(
                f"error: --build-cert: the certificate has no "
                f"{', '.join(missing)}, so every obligation it carries is "
                "`sorry`. With --infer this means inference produced none."
            )
    print(f".. Building the certificate in {project_dir}")
    try:
        build_certificate(project_dir)
    except LakeBuildError as e:
        raise SystemExit(f"error: --build-cert: {e}") from e


def hammer_only(hammer_file: str) -> None:
    """`--hammer-file`: write ZerothHammer.lean and return. No module needed."""
    out = Path(hammer_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_zeroth_hammer_lean())
    except OSError as e:
        raise SystemExit(f"error: --hammer-file: {e}") from e
    print(f"Wrote {out}")


def standalone(cert_file: str, module, lean_cert) -> None:
    """`--cert-file`: a self-contained certificate and its sibling encodings.

    No project, so no route: the predicates are the ones supplied, which is
    why every route that reads or writes a project is refused alongside it.
    """
    out = Path(cert_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_standalone_cert_lean(module, lean_cert))
    except (Refused, OSError) as e:
        raise SystemExit(f"error: --cert-file: {e}") from e
    print(f"Wrote standalone certificate: {out}")
    # The same encoding table the project route walks, so these files and
    # the ones in a generated project carry the same headers and the same
    # imports. The functional encoding is already inlined in the standalone
    # certificate above, which is the base they import.
    m2l = ModuleToLean4(module)
    layout = Layout(base=out.stem, flat=True, directory=out.parent)
    for enc in ENCODINGS:
        if not enc.cert_file:
            continue
        print(f"Wrote {enc.title.lower()}: {write_encoding(enc, m2l, layout, out.stem)}")


def generate(settings: Settings, module, ctx: LeanContext,
             lean_cert: "LeanCertificate | None",
             source: "inputs.SourceText") -> Path:
    """The Lean project, with the certificate as it stands before inference."""
    print(".. Generating lean code")
    try:
        return create_project(
            output_dir=settings.output_dir,
            module=module,
            project_name=settings.project_name,
            executable=settings.executable,
            certificate=certificate_files(ctx, lean_cert),
            source=source,
            lakefile=(
                settings.route.lakefile(
                    settings.route_config, settings.project_name)
                if settings.route is not None and settings.route.lakefile
                else None
            ),
            # `Certificate/` is globbed by the build, so a copy of a file a
            # route writes, left by an earlier run in this `-o`, would be
            # built against this run's module. Directories are left alone:
            # nothing globs them until a route declares a lean_lib for them.
            remove=owned_files(),
            ctx=ctx,
        )
    except (Refused, OSError) as e:
        raise SystemExit(f"error: {e}") from e
