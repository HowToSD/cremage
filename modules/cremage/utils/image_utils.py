"""
Utility functions related to handling images
"""
import io
import os
import time
from typing import List, Dict
import json

from einops import rearrange
import torch
import numpy as np
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import cv2 as cv

from .misc_utils import extract_embedding_filenames

def tensor_to_pil_image(tzo_image: torch.tensor):
    """

    Args:
        tzo_image: Torch tensor in (c, h, w) with values in [0, 1].
            tzo stands for tensor, zero, one
    Returns:
        Tuple of Image in PIL RGB format and torch device name that the tensor is on.
    """
    image = rearrange(tzo_image, "c h w -> h w c")
    pil_image = Image.fromarray(
        (image.detach().cpu().numpy() * 255.0).astype(np.uint8))
    return pil_image, str(tzo_image.device)


def pil_image_to_tensor(pil_image: torch.tensor, device=None, half=True):
    """

    Args:
        pil_image: Image in PIL RGB format
        half: Returns a float16 tensor if True. Else returns a float32 tensor.
    Returns:
        Torch tensor in (c, h, w) with values in [0, 1].
    """
    image = torch.tensor(np.asarray(pil_image).astype(np.float32))
    t = rearrange(image, "h w c -> c h w") / 255.0
    if half:
        t = t.half()
    if device is None:
        if torch.cuda.is_available():
            t = t.to("cuda")
    else:
        t = t.to(device)
    return t


def put_watermark(img, wm_encoder=None):
    """
    Add a watermark to the image.

    TODO: Consolidate this with SD1.5

    Args:
        img (Image): PIL-format image
        wm_encoder (WaterMarkEncoder)
    """
    if wm_encoder is not None:
        img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def save_torch_tensor_as_image_with_watermark(
        opt,
        save_dir: str,
        tzo_img: torch.tensor,
        meta:Dict=None,
        meta_attribute_name:str="generation_data",
        file_number:int=None,
        wm_encoder=None) -> Image:
    """
    Saves a PyTorch tensor as a file with meta data.

    Args:
        save_dir (str): Directory name of the file. If the directory does not exist, it will be created automatically.
        tzo_img (torch.Tensor): Torch tensor on GPU memory (e.g. CUDA) in the C, H, W format that needs to be saved.
                            It can be on the regular RAM as well.
                            The image has to be in float in [0, 1].
                            tzo stands for tensor, zero, one.
        meta (Dict): Meta data contained in a Python dictionary.
        meta_attribute_name (str): Name of the meta field to be stored in the file.
        file_number (int): A number used as a part of the file name.
        wm_encoder (WatermarkEncoder): Watermark encoder instance. If None, do not put watermark.
   
    Returns:
        The image in the PIL format.
    """
    # Check if directory exists
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    str_generation_params = json.dumps(meta)
    metadata = PngInfo()
    metadata.add_text(meta_attribute_name, str_generation_params)

    img = 255. * rearrange(tzo_img.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(img.astype(np.uint8))

    if wm_encoder:
        img = put_watermark(img, wm_encoder)

    time_str = time.time()
    file_name =  os.path.join(save_dir, f"{file_number:05}_{time_str}.png")

    img.save(file_name, pnginfo=metadata)
    img.info[meta_attribute_name] = str_generation_params


    # Save image as a sample image for the embeddings used
    # Extract embedding file name if any
    embedding_images_dir = opt.embedding_images_dir
    if embedding_images_dir:
        if os.path.exists(embedding_images_dir) is False:
            os.makedirs(embedding_images_dir)
    for prompt in [opt.prompt, opt.negative_prompt]:
        embedding_file_names = extract_embedding_filenames(prompt)
        for file_name in embedding_file_names:
            img.save(
                os.path.join(embedding_images_dir,
                        f"{file_name}.png"),
                pnginfo=metadata)

    return img


def save_torch_tensor_as_image(
        save_path: str,
        img: torch.tensor,
        meta:Dict=None,
        meta_attribute_name="generation_data") -> Image:
    """
    Saves a PyTorch tensor as a file with meta data.

    Args:
        save_path (str): Full path of the file. If the directory does not exist, it will be created automatically.
        img (torch.Tensor): Torch tensor on GPU memory (e.g. CUDA) in the C, H, W format that needs to be saved.
                            It can be on the regular RAM as well.
                            The image has to be in float in [0, 1].
        meta (Dict): Meta data contained in a Python dictionary.
        meta_attribute_name (str): Name of the meta field to be stored in the file.

    Returns:
        The image in the PIL format.
    """
    # Check if directory exists
    dir_name = os.path.dirname(save_path)
    if os.path.exists(dir_name) is False:
        os.makedirs(dir_name)

    str_generation_params = json.dumps(meta)
    metadata = PngInfo()
    metadata.add_text(meta_attribute_name, str_generation_params)

    img = 255. * rearrange(img.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(img.astype(np.uint8))

    img.save(save_path, pnginfo=metadata)
    img.info[meta_attribute_name] = str_generation_params
    return img


def bbox_for_multiple_of_64(width, height):
    def compute(edge_len):
        q = edge_len // 64  # 63->0, 64->1, 65->1
        target_len = 64 * q  # 0, 64, 64
        if edge_len % 64 > 0:     # T, F, T
            target_len += 64  # 64, 64, 128
        return target_len
    return compute(width), compute(height)

def pil_image_to_binary_pil_image(pil_image:Image) -> Image:
    """
    Converts the input RGB, RGBA or grayscale PIL image to a binary
    image that contains either 0 or 255 in rank 2 array.
    The image is converted to PIL after processing and is returned to the caller.

    Args:
        The input RGB, RGBA or grayscale PIL image
    Returns:
        The binary image (Image)
    """
    image_cv = np.asarray(pil_image)
    if len(image_cv.shape) != 2:
        gray_image = cv.cvtColor(image_cv, cv.COLOR_RGBA2GRAY)
    else:
        gray_image = image_cv
    _, binary_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
    return Image.fromarray(binary_image)


def resize_with_padding(image: Image,
                        *,
                        target_width:int=None,
                        target_height:int=None,
                        color:str="#ffffff",
                        return_bbox=False):
    """
    Image will be padded to maintain the original aspect ratio.
    """
    w, h = image.size
    if w == target_width and h == target_width:
        if return_bbox:
            # return (x1, y1, x2, y2) for the unpadded area
            return image, (0, 0, w, h)
        else:
            return image

    # Pad image
    base_image = Image.new('RGBA', (target_width, target_height), color)

    ratio_1 = target_width / w
    new_h = int(h * ratio_1)
    if new_h > target_height:  # won't fit
        ratio_2 = target_height / h
        new_w = int(w * ratio_2)
        if new_w > target_width:
            ValueError("Failed to resize")
        new_h = target_height
        padding_w = target_width - new_w
        if padding_w % 2 != 0:
            odd_pad = 1
        else:
            odd_pad = 0
        padding_h = 0
        padding_x = int(padding_w/2) + odd_pad
        padding_y = int(padding_h/2)
    else:
        new_w = target_width
        padding_h = target_height - new_h
        if padding_h % 2 != 0:
            odd_pad = 1
        else:
            odd_pad = 0
        padding_w = 0
        padding_x = int(padding_w/2)
        padding_y = int(padding_h/2) + odd_pad

    resized_image = image.resize((int(new_w), int(new_h)), resample=PIL.Image.LANCZOS)

    base_image.paste(resized_image, (padding_x, padding_y))
    if return_bbox:
        # return (x1, y1, x2, y2) for the unpadded area
        return base_image, (padding_x, padding_y, padding_x + new_w, padding_y + new_h)
    else:
        return base_image


def get_single_bounding_box_from_grayscale_image(cv_img: np.ndarray):
    """
    Detects a single bounding box that encloses all connected regions in the input grayscale image.

    Threshold is applied in this method to each pixel to ensure that the value only contains either:
    0:   Not masked (not to be inpainted)
    255: Masked (to be inpainted)
    If pixel value <= 127: 0
    else:                 255

    Args:
        cv_img (numpy.ndarray): Rank 2 OpenCV image. Each pixel can contain 0-255 in uint8.
    Returns:
        A tuple of coordinates for the single bounding box (left, top, width, height).
        If no region is found, None is returned.
    """
    if len(cv_img.shape) != 2:
        raise ValueError("Invalid image format. The input has to be a rank 2 OpenCV image or numpy array")

    _, binary_mask = cv.threshold(cv_img, 127, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(binary_mask,
                                  cv.RETR_EXTERNAL,  # Grab the outside contour if nested. Ignore contour inside
                                  cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # Calculate the combined bounding box that covers all contours
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    if x_min == float('inf') or y_min == float('inf') or x_max == float('-inf') or y_max == float('-inf'):
        return None

    width = x_max - x_min
    height = y_max - y_min

    return (x_min, y_min, width, height)


def get_bounding_boxes_from_grayscale_image(cv_img: np.ndarray):
    """
    Detects the list of bounding boxes that surround connected regions in the input grayscale image.

    Threshold is applied in this method to each pixel to ensure that the value only contains either:
    0:   Not masked (not to be inpainted)
    255: Masked (to be inpainted)
    If pixel value <= 127: 0
    else:                 255

    Args:
        cv_img (numpy.ndarray): Rank 2 OpenCV image. Each pixel can contain 0-255 in uint8.
    Returns:
        List[(left, top, width, height), ...] of bounding boxes.  If no region is found,
        an empty list is returned.
    """
    if cv_img.shape != 2:
        ValueError("Invalid image format. The input has to be a rank 2 OpenCV image or numpy array")

    _, binary_mask = cv.threshold(cv_img, 127, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(binary_mask,
                                  cv.RETR_EXTERNAL,  # Grab the outside contour if nested. Ignore contour inside
                                  cv.CHAIN_APPROX_SIMPLE)
    
    # Calculate bounding boxes: (x, y, w, h) for each contour
    bounding_boxes = [cv.boundingRect(contour) for contour in contours]
    return bounding_boxes


def get_bounding_boxes(pil_img):
    """
    Detects the list of bounding boxes that surround connected regions in the input RGBA image.

    Args:
        pil_image (Image): RGBA PIL Image. Alpha contains the mask value.
            Alpha channel value > 127 is considered as a mask.  For example,
            the area to be inpainted.
    Returns:
        List[(x, y, width, height), ...]: List of bounding boxes.
    """

    # Convert PIL image to OpenCV format
    img = np.array(pil_img)

    # Extract the alpha channel as the mask (assuming RGBA)
    # This is a rank 3 grayscale image [0, 255]
    _, _, _, alpha = cv.split(img)
    
    return get_bounding_boxes_from_grayscale_image(alpha)


def get_png_paths(image_dir: str) -> List[str]:
    """
    Retrieves a list of paths to PNG images within a specified directory, sorted by their creation time (oldest first).

    Args:
        image_dir (str): The directory path from which to retrieve PNG image paths.

    Returns:
        List[str]: A list of file paths to PNG images, sorted by creation time.

    Example::
        >>> image_dir = "/path/to/images"
        >>> image_paths = get_png_paths(image_dir)
        >>> print(image_paths)    
    """
    if image_dir is None:
        return []

    image_paths = os.listdir(image_dir)
    
    # Select png only
    # image_paths = [os.path.join(image_dir, f) for f in image_paths if f.endswith(".png")]
    image_paths = [os.path.join(image_dir, f) for f in image_paths 
                   if f.endswith(".png") or f.endswith(".webp") or f.endswith(".jpg")]

    # Sort the files by creation time (newest first)
    image_paths.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return image_paths


def pil_image_to_gtk_image(pil_img: Image):
    """
    Convert a PIL Image to a Gtk.Image.

    Args:
        pil_image (Image): The input PIL image object.

    Returns:
        Image: A GTK image.  
    """
    pixbuf = pil_image_to_pixbuf(pil_img)
    
    # Create a Gtk.Image and set the Pixbuf
    gtk_image = Gtk.Image.new_from_pixbuf(pixbuf)
    
    return gtk_image


def pil_image_from_gtk_image(gtk_image: Gtk.Image) -> Image:
    """
    Convert a Gtk.Image into a PIL Image.

    Args:
        gtk_image (Gtk.Image): The GTK image object to convert.

    Returns:
        Image: The converted PIL Image object.

    Raises:
        ValueError: If the Gtk.Image does not have a Pixbuf associated with it.
    """
    pixbuf = gtk_image.get_pixbuf()
    if pixbuf is None:
        raise ValueError("The Gtk.Image does not have a Pixbuf.")
    
    # Convert GdkPixbuf.Pixbuf to a numpy array
    width, height = pixbuf.get_width(), pixbuf.get_height()
    dtype = np.uint8  # GdkPixbuf.Pixbuf uses 8-bit channels
    channels = 3 if pixbuf.get_has_alpha() else 4  # RGB or RGBA depending on alpha
    pixels = np.frombuffer(pixbuf.get_pixels(), dtype=dtype)
    pixels = pixels.reshape((height, width, channels))

    # Convert numpy array to PIL Image
    mode = 'RGBA' if pixbuf.get_has_alpha() else 'RGB'
    pil_img = Image.fromarray(pixels, mode=mode)

    return pil_img


def pil_image_to_pixbuf(pil_img: Image.Image) -> GdkPixbuf.Pixbuf:
    """
    Converts a PIL Image object to a GdkPixbuf.Pixbuf object.

    This conversion allows the PIL Image to be used in GTK applications, 
    where GdkPixbuf is the standard format for images.

    Args:
        pil_img (PIL.Image.Image): The PIL Image object to convert.

    Returns:
        GdkPixbuf.Pixbuf: The resulting pixbuf object.

    Example:
        >>> from PIL import Image
        >>> pil_img = Image.open("example.png")
        >>> pixbuf = pil_image_to_pixbuf(pil_img)
    """
    # Save the PIL Image to a bytes buffer in PNG format
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)  # Seek to the start of the buffer
    
    # Load the image buffer into a GdkPixbuf
    loader = GdkPixbuf.PixbufLoader.new_with_type('png')
    loader.write(buf.getvalue())
    loader.close()
    pixbuf = loader.get_pixbuf()
    
    return pixbuf


def pil_image_from_pixbuf(pixbuf):
    """Converts a GdkPixbuf to a PIL Image."""
    data = pixbuf.get_pixels()
    w = pixbuf.get_width()
    h = pixbuf.get_height()
    stride = pixbuf.get_rowstride()
    mode = "RGB"
    if pixbuf.get_has_alpha():
        mode = "RGBA"
    # Create a PIL image from the pixbuf data
    image = Image.frombytes(mode, (w, h), data, "raw", mode, stride)
    return image


def display_pil_image_from_mask_pil_image(image: Image):
    """Converts black pixels to transparent in a PIL input image"""
    img = image.convert("RGBA")
    data = np.array(img)  # Convert to NumPy array to manipulate

    # Identify dark pixels
    black_pixels = (data[:,:,0] < 128) & (data[:,:,1] < 128) & (data[:,:,2] < 128)
    data = np.full(data.shape, 255, dtype=np.uint8)
    data[black_pixels] = (0, 0, 0, 0)  # Set RGBA values for identified pixels (transparent)

    return Image.fromarray(data, 'RGBA')


def display_pil_image_to_mask_pil_image(image: Image):
    """Converts transparent pixels to black in a PIL input image"""
    if image.mode != 'RGBA':
        raise ValueError("Image must be in RGBA mode to identify transparent pixels")
    
    img = np.array(image)  # Convert to NumPy array to manipulate

    # Identify transparent pixels (A=0)
    transparent_pixels = img[:,:,3] == 0
    img[transparent_pixels] = (0, 0, 0, 255)  # Set identified pixels to black (R=0, G=0, B=0, A=255)

    return Image.fromarray(img, 'RGBA')


def load_resized_pil_image(image_path:str, target_size:int=128) -> Image:
    """
    Load an image from a specified path and resize it so that its longer edge equals `target_size`,
    maintaining the aspect ratio of the original image.

    Args:
        image_path (str): The file path to the image to be loaded.
        target_size (int, optional): The target size for the longer edge of the image. Defaults to 128.

    Returns:
        Image: A PIL Image object of the resized image.
    """    
    pil_image = Image.open(image_path)
    w, h = pil_image.size 

    # Resize the image maintaining aspect ratio
    if h > w:
        w = int(target_size * w / h)
        h = target_size
    else:
        h = int(target_size * h / w)
        w = target_size

    pil_image = pil_image.resize((w, h))  # Resize takes a tuple (new_width, new_height)
    return pil_image


def resize_pil_image(pil_image:Image, target_size:int=128) -> Image:
    """
    Resize the PIL image so that its longer edge equals `target_size`,
    maintaining the aspect ratio of the original image.

    Args:
        pil_image (Image): The PIL image.
        target_size (int, optional): The target size for the longer edge of the image. Defaults to 128.

    Returns:
        Image: A PIL Image object of the resized image.
    """    
    w, h = pil_image.size 

    # Resize the image maintaining aspect ratio
    if h > w:
        w = int(target_size * w / h)
        h = target_size
    else:
        h = int(target_size * h / w)
        w = target_size

    return pil_image.resize((w, h), resample=PIL.Image.LANCZOS)  # Resize takes a tuple (new_width, new_height)

