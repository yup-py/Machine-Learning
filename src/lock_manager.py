import os
import json
from datetime import date

# Simple path to store lock files
LOCKS_DIR = "locks"

class StepAlreadyDone(Exception):
    """Raised if the step was already finished today."""
    pass

class LockManager:
    def __init__(self, step_name, force=False):
        self.step_name = step_name
        self.force = force
        self.today = date.today().isoformat()
        self.path = self._get_path()

    def _get_path(self):
        folder = os.path.join(LOCKS_DIR, self.step_name)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{self.today}.json")

    def __enter__(self):
        # If not forcing, check if we already finished today
        if not self.force and os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                if data.get("status") == "done":
                    raise StepAlreadyDone(f"Step '{self.step_name}' already finished today.")
        
        print(f"[lock] Starting step: {self.step_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "done" if exc_type is None else "failed"
        with open(self.path, "w") as f:
            json.dump({"status": status, "step": self.step_name}, f)
        
        if status == "done":
            print(f"[lock] Finished step: {self.step_name}")
        else:
            print(f"[lock] Step failed: {self.step_name} - {exc_val}")