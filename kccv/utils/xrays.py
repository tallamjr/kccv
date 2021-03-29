import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_xray(path, voi_lut=True, fix_monochrome=True):
    """
    Refs
    ----
    - https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> %matplotlib inline
    >>> img = read_xray('../input/vinbigdata-chest-xray-abnormalities-detection/train/0108949daa13dc94634a7d650a05c0bb.dicom')
    >>> plt.figure(figsize = (12,12))
    >>> plt.imshow(img, 'gray')

    """
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data
