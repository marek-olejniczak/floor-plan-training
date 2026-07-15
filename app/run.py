from pathlib import Path
import sys

app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))

print("Importing app...")
import app as app_module
print(f"Model loaded: {app_module.model}")
print("Starting server on port 5000...")
app_module.app.run(host="0.0.0.0", port=5000, debug=True)
