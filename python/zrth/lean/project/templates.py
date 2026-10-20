from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pathlib import Path

from zrth.lean.project.names import LINTERS_OFF

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"
PROJECT_TEMPLATES_DIR = TEMPLATES_DIR / "project"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)
# Every generated file says it after its imports; see `LINTERS_OFF`.
env.globals["linters_off"] = LINTERS_OFF


def render(template_path: str, **context: object) -> str:
    """Render a template relative to the templates/ directory."""
    return env.get_template(template_path).render(**context)
