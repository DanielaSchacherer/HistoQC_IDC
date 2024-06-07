import logging
import os
from skimage import io, img_as_ubyte
from distutils.util import strtobool
from skimage import color
import numpy as np

import matplotlib.pyplot as plt
from histoqc.BaseImage import BaseImage
import highdicom as hd 
from pydicom import dcmread

def blend2Images(img, mask):
    if (img.ndim == 3):
        img = color.rgb2gray(img)
    if (mask.ndim == 3):
        mask = color.rgb2gray(mask)
    img = img[:, :, None] * 1.0  # can't use boolean
    mask = mask[:, :, None] * 1.0
    out = np.concatenate((mask, img, mask), 2)
    return out

def saveFinalMaskToDicomSeg(mask: np.ndarray[bool], s: BaseImage) -> None:
    binary_mask = img_as_ubyte(mask) 
    # printing testing 
    message = f"bindim {binary_mask.ndim} {binary_mask}"
    logging.warning(message)
    s["warnings"].append(message)

    source_image = dcmread(os.path.join(s['dir'], s['filename']))
    test_mask = np.zeros(
    (
        source_image.TotalPixelMatrixRows,
        source_image.TotalPixelMatrixColumns
    ),
    dtype=np.uint8,
    )
    test_mask[38:43, 5:41] = 1
    message = f"bindim {test_mask.ndim} {test_mask}"
    logging.warning(message)
    s["warnings"].append(message)

    property_category = hd.sr.CodedConcept("91723000", "SCT", "Anatomical Structure")
    property_type = hd.sr.CodedConcept("84640000", "SCT", "Nucleus")
    segment_descriptions = [
        hd.seg.SegmentDescription(
            segment_number=1,
            segment_label='Segment #1',
            segmented_property_category=property_category,
            segmented_property_type=property_type,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        )
    ]

    
    seg = hd.seg.Segmentation(
        source_images=[source_image],
        pixel_array=test_mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Slide Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
        tile_pixel_array=True,
    )


def saveFinalMask(s, params):
    logging.info(f"{s['filename']} - \tsaveUsableRegion")

    mask = s["img_mask_use"]
    for mask_force in s["img_mask_force"]:
        mask[s[mask_force]] = 0

    # printing testing 
    message = f"{img_as_ubyte(mask).ndim}"
    logging.warning(message)
    s["warnings"].append(message)

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_mask_use.png", img_as_ubyte(mask))
    saveFinalMaskToDicomSeg(mask, s)

    if strtobool(params.get("use_mask", "True")):  # should we create and save the fusion mask?
        img = s.getImgThumb(s["image_work_size"])
        out = blend2Images(img, mask)
        io.imsave(s["outdir"] + os.sep + s["filename"] + "_fuse.png", img_as_ubyte(out))

    return


def saveAssociatedImage(s, key:str, dim:int):
    logging.info(f"{s['filename']} - \tsave{key.capitalize()}")
    osh = s["os_handle"]

    if not key in osh.associated_images:
        message = f"{s['filename']}- save{key.capitalize()} Can't Read '{key}' Image from Slide's Associated Images"
        logging.warning(message)
        s["warnings"].append(message)
        return
    
    # get asscociated image by key
    associated_img = osh.associated_images[key]
    (width, height)  = associated_img.size

    # calulate the width or height depends on dim
    if width > height:
        h = round(dim * height / width)
        size = (dim, h)
    else:
        w = round(dim * width / height)
        size = (w, dim)
    
    associated_img = associated_img.resize(size)
    associated_img = np.asarray(associated_img)[:, :, 0:3]
    io.imsave(f"{s['outdir']}{os.sep}{s['filename']}_{key}.png", associated_img)

def saveMacro(s, params):
    dim = params.get("small_dim", 500)
    saveAssociatedImage(s, "macro", dim)
    return
    
def saveMask(s, params):
    logging.info(f"{s['filename']} - \tsaveMaskUse")
    suffix = params.get("suffix", None)
    
    # check suffix param
    if not suffix:
        msg = f"{s['filename']} - \tPlease set the suffix for mask use."
        logging.error(msg)
        return

    # save mask
    io.imsave(f"{s['outdir']}{os.sep}{s['filename']}_{suffix}.png", img_as_ubyte(s["img_mask_use"]))

def saveThumbnails(s, params):
    logging.info(f"{s['filename']} - \tsaveThumbnail")
    # we create 2 thumbnails for usage in the front end, one relatively small one, and one larger one
    img = s.getImgThumb(params.get("image_work_size", "1.25x"))
    io.imsave(s["outdir"] + os.sep + s["filename"] + "_thumb.png", img)

    img = s.getImgThumb(params.get("small_dim", 500))
    io.imsave(s["outdir"] + os.sep + s["filename"] + "_thumb_small.png", img)
    return
