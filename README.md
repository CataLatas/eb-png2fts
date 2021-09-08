# eb-png2fts
Convert image files into [CoilSnake](https://github.com/pk-hack/CoilSnake/)-compatible .FTS files for Earthbound rom-hacking

## Usage
```
python eb_png2fts.py [-h] IN_IMAGE [IN_IMAGE ...] -t TILESET_ID -o OUTPUT

-h, --help                    Show a help message
-t, --tileset-id TILESET_ID   Specify the tileset ID (in range 0-19 inclusive)
-o, --output OUTPUT           Output FTS file
IN_IMAGE                      Input image file (accepts multiple images)
```

Example:
`python eb_png2fts.py maps/Twoson*.png -t 2 -o ~/CoilSnake/MyProject/Tilesets/02.fts`

## TODO
- Support for foreground tiles
- Support for surface/collision flags
- Support for inserting chunks into sectors via the `map_sectors.yml` and `map_tiles.map` files
- Better palette generation. Currently, converting an image with around 40 colors fails
- Optimization. It takes a while to generate a tileset, especially if the images are big and there are many unique tiles
