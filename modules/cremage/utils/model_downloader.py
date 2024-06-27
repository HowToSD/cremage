"""
Dowloads the unblur face model.

Test code path: test/unblur_face/model_downloader_test.py

Copyright 2024 Hideyuki Inada.  All rights reserved.
"""
import os
import logging
import shutil

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def download_model_if_not_exist(local_model_dir, repo_id, file_name, cache_dir=None):
    """
    Downloads a model file from a Hugging Face repository.

    Args:
        repo_id (str): The ID of the Hugging Face repository (e.g., "username/repo_name").
        filename (str): The name of the file to download (e.g., "foo.pth").
        cache_dir (str, optional): Directory where the file will be cached. Defaults to None.

    Returns:
        str: The path to the downloaded file.
    """
    local_model_path = os.path.join(local_model_dir, file_name)
    if os.path.exists(local_model_path) is False:
        logger.info(f"Downloading {file_name} from Hugging Face repo {repo_id}")
        from huggingface_hub import hf_hub_download
        file_path = hf_hub_download(repo_id=repo_id, filename=file_name, cache_dir=cache_dir)
        if os.path.exists(file_path) is False:
            raise ValueError(f"{file_path} not found after model download")

        # Resolve the symlink to the actual file path
        if os.path.islink(file_path):
            file_path = os.path.realpath(file_path)

        if os.path.exists(local_model_dir) is False:
            os.makedirs(local_model_dir)

        shutil.copy(file_path, local_model_path)
        logger.info(f"Model saved in {local_model_path}")
        
    return local_model_path

