import argparse
from map_processing import image_as_numpy, process_image, save_numpy_as_img

parser = argparse.ArgumentParser(description="Applies the map cutout algorithm to a single image and saves it.")

parser.add_argument("--source_image", type=str, required=True, help="The path to the source image.")
parser.add_argument("--target_image", type=str, required=True, help="The path to the target image.")
parser.add_argument("--rows", type=int, required=True, help="The number of map component rows.")
parser.add_argument("--cols", type=int, required=True, help="The number of map component columns.")
parser.add_argument("--processing_height", type=int, default="600", help="The image processing height.")
parser.add_argument("--n_considered_points", type=int, default="100", help="The number of points utilized during "
                                                                           "regression.")

args = parser.parse_args()


img_ary = image_as_numpy(args.source_image, max_height=args.processing_height)
processed_image = process_image(
    img_ary,
    rows=args.rows,
    columns=args.cols,
    regression_considered_points=args.n_considered_points,
)
save_numpy_as_img(processed_image, args.target_image)

print("Done.")