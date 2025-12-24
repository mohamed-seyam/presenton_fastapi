import os
from utils.get_env import get_app_data_directory_env


def get_images_directory():
    images_directory = os.path.join(get_app_data_directory_env(), "images")
    os.makedirs(images_directory, exist_ok=True)
    return images_directory


def get_exports_directory():
    export_directory = os.path.join(get_app_data_directory_env(), "exports")
    os.makedirs(export_directory, exist_ok=True)
    return export_directory

def get_uploads_directory():
    uploads_directory = os.path.join(get_app_data_directory_env(), "uploads")
    os.makedirs(uploads_directory, exist_ok=True)
    return uploads_directory


def convert_file_path_to_url(file_path: str) -> str:
    """
    Convert a local file path to an HTTP URL that can be accessed via the mounted static files.

    Args:
        file_path: Absolute path to the file (e.g., '/path/to/app_data/images/file.jpg')

    Returns:
        HTTP URL path (e.g., '/app_data/images/file.jpg')
    """
    # If already a HTTP URL, return as-is
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return file_path

    # If already starts with /app_data/ or /static/, it's already a URL path
    if file_path.startswith("/app_data/") or file_path.startswith("/static/"):
        return file_path

    # Get the app_data directory
    app_data_dir = get_app_data_directory_env()

    # Make sure the file path is absolute
    abs_file_path = os.path.abspath(file_path)
    abs_app_data_dir = os.path.abspath(app_data_dir)

    # Check if file is in app_data directory
    if abs_file_path.startswith(abs_app_data_dir):
        # Get the relative path from app_data directory
        rel_path = os.path.relpath(abs_file_path, abs_app_data_dir)
        # Convert to URL path
        url_path = f"/app_data/{rel_path.replace(os.sep, '/')}"
        return url_path

    # If not in app_data, return the original path
    return file_path
