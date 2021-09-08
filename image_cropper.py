# TODO: Turn this into a generator?

def get_tiles(image, tile_size):
    """Splits an image into multiple tiles of a certain size"""

    tile_images = []
    x = 0
    y = 0
    while y < image.height:
        im_tile = image.crop((x, y, x+tile_size, y+tile_size))
        tile_images.append(im_tile)

        if x < image.width - tile_size:
            x += tile_size
        else:
            x = 0
            y += tile_size

    return tile_images
