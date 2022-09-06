import argparse

import image_cropper
import palettepacker

from PIL import Image, ImageOps

HEX_DIGITS = '0123456789abcdef'
BASE32_DIGITS = HEX_DIGITS + 'ghijklmnopqrstuv'


class PaletteError(Exception):
    """Exception class for palette related errors"""


class EbPalette:
    """Represents a color palette of 96 colors (6 subpalettes of 15 colors)"""

    def __init__(self, backdrop=(0, 248, 0)):
        self.backdrop = backdrop
        self.subpalettes = [[] for _ in range(6)]

    def to_image(self):
        """Returns an image representation of the palette"""
        im = Image.new('RGB', (16, 6), self.backdrop)
        for y, colors in enumerate(self.subpalettes):
            for x in range(15):
                try:
                    value = colors[x]
                except IndexError:
                    value = self.backdrop

                im.putpixel((x+1, y), value)

        # Now resize it to 8x size
        im = im.resize((16*8, 6*8), resample=Image.NEAREST)
        return im

    def fts_string(self, area_id):
        """Returns the .fts string representation of the palette"""
        palette_str = BASE32_DIGITS[area_id] + '0' # Area, palette 0
        for colors in self.subpalettes:
            palette_str += '000' # Color 0 is always transparent, fixed to 000

            for i in range(15):
                try:
                    r, g, b = colors[i]
                except IndexError:
                    r, g, b = self.backdrop

                palette_str += BASE32_DIGITS[r // 8]
                palette_str += BASE32_DIGITS[g // 8]
                palette_str += BASE32_DIGITS[b // 8]

        return palette_str


class EbTile:
    """Represents an 8x8 tile"""

    def __init__(self, data, palette, palette_row, index=0, is_flipped_h=False, is_flipped_v=False):
        self.data = data
        self.palette = palette
        self.palette_row = palette_row
        self.index = index
        self.is_flipped_h = is_flipped_h
        self.is_flipped_v = is_flipped_v

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                (self.data, self.palette, self.palette_row, self.index, self.is_flipped_h, self.is_flipped_v) ==
                (other.data, other.palette, other.palette_row, self.index, self.is_flipped_h, self.is_flipped_v))

    def __hash__(self):
        return hash((self.data, self.palette, self.palette_row, self.index, self.is_flipped_h, self.is_flipped_v))

    @property
    def is_flipped(self):
        """Returns True if the tile is flipped either horizontally or vertically"""
        return self.is_flipped_h or self.is_flipped_v

    @property
    def is_flipped_hv(self):
        """Returns True if the tile is flipped both horizontally and vertically"""
        return self.is_flipped_h and self.is_flipped_v

    def to_image(self):
        """Returns an image representation of the tile"""
        image = Image.new('RGB', (8, 8))
        colors = self.palette[self.palette_row]
        for y, row in enumerate(self.data):
            for x, pixel in enumerate(row):
                image.putpixel((x, y), colors[pixel])

        return image

    def fts_string(self):
        """Returns the .fts string representation of the tile"""
        return ''.join(HEX_DIGITS[pixel] for pixel in self.data)


class EbChunk:
    """Represents a 32x32 chunk of 16 tiles with surface flag data"""

    def __init__(self, tiles, surface_flags):
        self.tiles = tiles
        self.surface_flags = surface_flags

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                (self.tiles, self.surface_flags) == (other.tiles, other.surface_flags))

    def to_image(self):
        """Returns an image representation of the chunk"""
        image = Image.new('RGB', (32, 32))

        x = 0
        y = 0
        for tile in self.tiles:
            im_tile = tile.to_image()
            image.paste(im_tile, (x, y))

            if x < 32 - 8:
                x += 8
            else:
                x = 0
                y += 8

        return image

    def fts_string(self):
        """Returns the .fts file string representation of the chunk"""
        s = ''
        for i, tile in enumerate(self.tiles):
            snes_tile = tile.index | (0x0800 + (tile.palette_row << 10))
            if tile.is_flipped_h:
                snes_tile |= 0x4000

            if tile.is_flipped_v:
                snes_tile |= 0x8000

            surface = self.surface_flags[i]
            s += f'{snes_tile:04x}{surface:02x}'

        return s


class EbTileset:
    """Represents a collection of unique 32x32 chunks and 8x8 tiles with a palette"""

    def __init__(self, tileset_id):
        if 0 > tileset_id > 19:
            raise ValueError('Tileset ID must be in range 0..19')

        # Data set prior to calling compute()
        self.chunk_tile_images = []
        self.tile_palettes = []

        # Data set by compute()
        self.tile_index = 0 # Index for next unique tile
        self.tileset_id = tileset_id
        self.chunks = []
        self.tiles = []
        self.palette = EbPalette()
        self.tile_dict = dict()
        self._chunk_image_cache = set()

    def append_from_image(self, image):
        """Adds unique chunks and tiles from an image into the tileset"""
        chunk_images = image_cropper.get_tiles(image, tile_size=32)

        for im_chunk in chunk_images:
            if im_chunk.tobytes() in self._chunk_image_cache:
                continue

            self._chunk_image_cache.add(im_chunk.tobytes())

            tile_images = image_cropper.get_tiles(im_chunk, tile_size=8)
            self.chunk_tile_images.append(tile_images)
            for im_tile in tile_images:
                colors = im_tile.getcolors(15) # (count, (r,g,b))
                if colors is None:
                    raise PaletteError('A single tile had more than 15 colors.')

                colors = [rgb for _, rgb in colors] # Discard pixel count
                self.tile_palettes.append(colors)

    def compute(self):
        # Use palettepacker library to perform better packing of
        # palettes into subpalettes
        self.palette.subpalettes, subpalette_map = \
            palettepacker.tilePalettesToSubpalettes(self.tile_palettes)

        for chunk_idx, tile_images in enumerate(self.chunk_tile_images):
            chunk_tiles = []
            for tile_idx, im_tile in enumerate(tile_images):
                im_tile_h = im_tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                im_tile_v = im_tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                im_tile_hv = im_tile_h.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                tile_hash = im_tile.tobytes()
                tile_h_hash = im_tile_h.tobytes()
                tile_v_hash = im_tile_v.tobytes()
                tile_hv_hash = im_tile_hv.tobytes()
                if tile_hash not in self.tile_dict:
                    palette_row = subpalette_map[chunk_idx * 16 + tile_idx]
                    subpalette = self.palette.subpalettes[palette_row]
                    image_data = im_tile.getdata()
                    tile_data = tuple(subpalette.index(c)+1 for c in image_data)

                    tile = EbTile(tile_data, self.palette, palette_row, index=self.tile_index)
                    tile_h = EbTile(tile_data, self.palette, palette_row, index=self.tile_index, is_flipped_h=True)
                    tile_v = EbTile(tile_data, self.palette, palette_row, index=self.tile_index, is_flipped_v=True)
                    tile_hv = EbTile(tile_data, self.palette, palette_row, index=self.tile_index, is_flipped_h=True, is_flipped_v=True)
                    self.tile_index += 1

                    self.tile_dict[tile_hash] = tile
                    self.tile_dict[tile_h_hash] = tile_h
                    self.tile_dict[tile_v_hash] = tile_v
                    self.tile_dict[tile_hv_hash] = tile_hv
                else:
                    tile = self.tile_dict[tile_hash]

                if tile not in self.tiles:
                    self.tiles.append(tile)

                chunk_tiles.append(tile)

            chunk = EbChunk(chunk_tiles, [0x00] * 16) # Default surface flags to zeros for now...
            if chunk not in self.chunks:
                self.chunks.append(chunk)

    def to_fts(self, filepath):
        """Writes a .fts file containing the data for the tileset"""
        if len(self.tiles) >= 512:
            # TODO Custom exception class
            raise RuntimeError('Oops too many unique tiles!')

        if len(self.chunks) >= 1024:
            # TODO Custom exception class
            raise RuntimeError('Oops too many unique chunks!')

        with open(filepath, 'w', encoding='utf-8', newline='\n') as fts_file:
            # First, 512 tile graphics definition:
            # All 64 pixels for the BACKGROUND tile. Each digit is an index into palette (0..F)
            # All 64 pixels for the FOREGROUND tile. Each digit is an index into palette (0..F)
            # (newline here)

            blank_graphics_str = '0' * 64
            for tile in self.tiles:
                fts_file.write(f'{tile.fts_string()}\n')
                fts_file.write(f'{blank_graphics_str}\n\n') # FOREGROUND is kept blank for now

            for i in range(len(self.tiles), 512):
                fts_file.write(f'{blank_graphics_str}\n')
                fts_file.write(f'{blank_graphics_str}\n\n')

            # Then, a newline followed by the palette information
            # AP(ppp(x16)(x6)), where:
            #   A = "area" (0..31)
            #   P = "palette" (0..7?)
            #   ppp = a single color in base32 (6 subpalettes of 16 colors each)
            palette_str = self.palette.fts_string(self.tileset_id)
            fts_file.write('\n')
            fts_file.write(f'{palette_str}\n')

            # Then, two newlines followed by 1024 "32x32 chunk" definitions:
            # ttttss(x16), where:
            #   t = tile (SNES format)
            #   s = surface flags.
            #   Note: Inexistant chunks use "000000"
            fts_file.write('\n\n')
            for chunk in self.chunks:
                fts_file.write(f'{chunk.fts_string()}\n')

            blank_chunk_str = '0' * 6*16
            for i in range(len(self.chunks), 1024):
                fts_file.write(f'{blank_chunk_str}\n')


def main(args):
    tileset = EbTileset(args.tileset_id)

    for path in args.input_files:
        with Image.open(path) as image:
            image = image.convert(mode='RGB') # Get rid of the alpha channel
            image = ImageOps.posterize(image, 5) # 5-bit color

            tileset.append_from_image(image)

    tileset.compute()

    print('Done!')
    print(f'{len(tileset.chunks)} chunks!')
    print(f'{len(tileset.tiles)} unique tiles!')

    tileset.to_fts(args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert image files into CoilSnake-compatible files for Earthbound hacking')

    parser.add_argument('input_files', metavar='IN_IMAGE', nargs='+', help='Input image files')
    parser.add_argument('-t', '--tileset-id', required=True, type=int, metavar='[0-19]', choices=range(20), help='Specify the tileset ID')
    parser.add_argument('-o', '--output', required=True, help='Output FTS file')
    args = parser.parse_args()

    from time import perf_counter

    start_time = perf_counter()

    main(args)

    elapsed = perf_counter() - start_time
    print(f'Time taken: {elapsed:.02f}s')
