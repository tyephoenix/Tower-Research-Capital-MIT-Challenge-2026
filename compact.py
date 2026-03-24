"""Generate the submission notebook using rep2nb."""
from rep2nb import convert

convert(
    repo_path=".",
    output="limestone_data_challenge_2026.ipynb",
    exclude=["answers", "compact.py"],
    include_pip_install=True,
)
