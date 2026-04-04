import io
import zipfile
from typing import Any, Dict


def create_framework_zip(data: Dict[str, Any]) -> bytes:
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("README.md", f"# AI Generated Playwright POM Framework\nModel: {data.get('model_used', 'Unknown')}\n")
        zip_file.writestr("requirements.txt", "pytest\npytest-playwright\n")
        zip_file.writestr("pytest.ini", "[pytest]\naddopts = --headed --browser chromium\n")

        for i, script in enumerate(data.get("scripts", []), 1):
            filename = f"tests/test_{i:03d}_generated.py"
            zip_file.writestr(filename, script)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
