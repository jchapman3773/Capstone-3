from PIL import Image
import os

def fix_orientation(filename):
    img = Image.open(filename)
    if hasattr(img, '_getexif'):
        exifdata = img._getexif()
        try:
            orientation = exifdata.get(274)
        except:
            # There was no EXIF Orientation Data
            orientation = 1
    else:
        orientation = 1

    if orientation is 1:    # Horizontal (normal)
        pass
    elif orientation is 2:  # Mirrored horizontal
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation is 3:  # Rotated 180
        img = img.rotate(180,expand=1)
    elif orientation is 4:  # Mirrored vertical
        img = img.rotate(180,expand=1).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation is 5:  # Mirrored horizontal then rotated 90 CCW
        img = img.rotate(-90,expand=1).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation is 6:  # Rotated 90 CCW
        img = img.rotate(-90,expand=1)
    elif orientation is 7:  # Mirrored horizontal then rotated 90 CW
        img = img.rotate(90,expand=1).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation is 8:  # Rotated 90 CW
        img = img.rotate(90,expand=1)

    return img

def save_imgs(dir):
    count = 0
    # TEST_IMAGE_PATHS = []
    filenames = os.listdir(dir)
    for f in filenames:
        count += 1
        path = os.path.join(dir,f)
        img = fix_orientation(path)
        img.save(path)

    print(f'{count} images resaved')

def list_orientations(dir):
    filenames = os.listdir(dir)
    for f in filenames:
        path = os.path.join(dir,f)
        img = Image.open(path)
        if hasattr(img, '_getexif'):
            exifdata = img._getexif()
            try:
                orientation = exifdata.get(274)
            except:
                # There was no EXIF Orientation Data
                orientation = 1
        else:
            orientation = 1
        print(f'{orientation} : {f}')

if __name__ == '__main__':
    dir = '../data/uploads'
    save_imgs(dir)
    # list_orientations(dir)
