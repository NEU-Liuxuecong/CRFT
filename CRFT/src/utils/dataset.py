import io
from loguru import logger
import struct
import cv2
import numpy as np
import h5py
import torch
from numpy.linalg import inv
import os
import os.path as osp


try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None

# --- DATA IO ---

def read_flow(path):
    with open(path,'rb') as f:
        magic=np.fromfile(f,np.float32,count=1)
        if magic!=202021.25:
            raise ValueError('Invalid .flo file')
        w=np.fromfile(f,np.int32,count=1)[0]
        h=np.fromfile(f,np.int32,count=1)[0]
        flow=np.fromfile(f,np.float32,count=2*w*h)
        flow=flow.reshape(h,w,2).transpose(2,0,1)#2 512 512
        flow=torch.from_numpy(flow).float()#-50+->+50+#前3个14 12.899452 13.798903
    return flow
        
        
def load_array_from_s3(
    path, client, cv_type,
    use_h5py=False,
):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.frombuffer(byte_str,np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data


def imread_gray(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
                else cv2.IMREAD_COLOR
    if str(path).startswith('s3://'):
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        image = cv2.imread(str(path), cv_type)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)

def read_image_rgb(path, resize=None, df=None, padding=False, augment_fn=None, np_random=None):
    """
    Read an image from path, convert it to RGB (3-channel), and apply transformations.
    This function handles both grayscale and color images, converting them to a 3-channel format.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found at {path}")

    # Convert to 3-channel RGB
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA/BGRA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- From here, the logic is similar to read_megadepth_gray ---
    if augment_fn is not None:
        image = augment_fn(image)

    # resize image
    if resize is not None and resize > 0:
        h, w = image.shape[:2]
        scale = resize / max(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

    # padding
    if padding:
        h, w = image.shape[:2]
        h_pad = df - h % df if h % df != 0 else 0
        w_pad = df - w % df if w % df != 0 else 0
        image = np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)), 'constant', constant_values=0)

    image = torch.from_numpy(image).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
    
    # The mask and scale might need adjustment based on your model's needs.
    # Here we assume the whole image is valid.
    mask = torch.ones_like(image[0], dtype=torch.bool)
    if resize is not None and resize > 0:
        scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)
    else:
        scale = torch.tensor([1.0, 1.0], dtype=torch.float)

    return image, mask, scale

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_center(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:])
    mask = None
    h, w = inp.shape[-2:]
    top = (pad_size - h) // 2
    left = (pad_size - w) // 2
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[top:top+h, left:left+w] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[top:top+h, left:left+w] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, top:top+h, left:left+w] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, top:top+h, left:left+w] = True
    else:
        raise NotImplementedError()
    return padded, mask

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_center(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask)

    return image, mask, scale


def read_megadepth_depth(path, pad_to=None):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
    else:
        depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_center(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


# --- ScanNet ---

def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


def read_scannet_depth(path):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(str(path), SCANNET_CLIENT, cv2.IMREAD_UNCHANGED)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]


# --- VisTir ---

def read_vistir_gray(path, cam_K, dist, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        cam_K (3, 3): camera matrix
        dist (8): distortion coefficients
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn, client=None)

    h,  w = image.shape[:2]
    # update camera matrix
    new_K, roi = cv2.getOptimalNewCameraMatrix(cam_K, dist, (w,h), 0, (w,h))
    # undistort image
    image = cv2.undistort(image, cam_K, dist, None, new_K)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_center(image, pad_to, ret_mask=True)
        mask = torch.from_numpy(mask)
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized

    return image, mask, scale, new_K

# --- PRETRAIN ---

def read_pretrain_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w) gray scale image
        image_norm (torch.tensor): (1, h, w) normalized image
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]   
        image_mean (torch.tensor): (1, 1, 1, 1)
        image_std (torch.tensor): (1, 1, 1, 1)
    """
    # read image
    image = imread_gray(path, augment_fn, client=None)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    image = image.astype(np.float32) / 255

    image_mean = image.mean()
    image_std = image.std()
    image_norm = (image - image_mean) / (image_std + 1e-6)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_center(image, pad_to, ret_mask=True)
        image_norm, _ = pad_center(image_norm, pad_to, ret_mask=False)
        mask = torch.from_numpy(mask)
    else:
        mask = None
    
    image_mean = torch.as_tensor(image_mean).float()[None,None,None]
    image_std = torch.as_tensor(image_std).float()[None,None,None]

    image = torch.from_numpy(image).float()[None]
    image_norm = torch.from_numpy(image_norm).float()[None]  # (h, w) -> (1, h, w) and normalized

    return image, image_norm, mask, scale, image_mean, image_std

