def get_tiles(image, tile_size):
    """Splits an image into multiple tiles of a certain size"""

    tile_images = []
    y = 0
    while y < image.height:
        x = 0
        while x < image.width:
            im_tile = image.crop((x, y, x+tile_size, y+tile_size))
            tile_images.append(im_tile)

            x += tile_size

        y += tile_size

    return tile_images
