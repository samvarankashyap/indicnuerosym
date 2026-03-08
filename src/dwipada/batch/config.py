"""Shared configuration loaders for batch processing modules.

Loads API keys and Vertex AI config from the project-level config.yaml.
"""

import sys
import yaml

from dwipada.paths import CONFIG_FILE


def load_api_key() -> str:
    """Load and return the Gemini API key from config.yaml."""
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    return config["api_key"]


def load_vertex_config() -> dict:
    """Load and return the Vertex AI configuration dict from config.yaml.

    Required fields under ``vertex``:
        project_id, location, gcs_bucket

    Optional fields:
        service_account_key, model
    """
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    vertex = config.get("vertex", {})
    required = ["project_id", "location", "gcs_bucket"]
    missing = [k for k in required if not vertex.get(k)]
    if missing:
        print(
            f"Error: Missing required fields in config.yaml under 'vertex': {missing}\n"
            "Add the following to config.yaml:\n"
            "  vertex:\n"
            "    project_id: your-gcp-project\n"
            "    location: us-central1\n"
            "    gcs_bucket: your-gcs-bucket\n"
            "    service_account_key: path/to/service_account.json  # optional\n"
            "    model: publishers/google/models/gemini-3-flash-preview  # optional",
            file=sys.stderr,
        )
        sys.exit(1)
    return vertex
