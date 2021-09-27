import math
import numpy as np

from PIL import Image
from skimage import filters
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, TheilSenRegressor
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import mode

import matplotlib.pyplot as plt

debug_img = None

REGRESSOR = LinearRegression
#REGRESSOR = HuberRegressor
#REGRESSOR = Ridge
#REGRESSOR = TheilSenRegressor

N_CONSIDERED_POINTS = 100  # Number of points considered for linear regression of lines
SEARCH_WIDTH = 10

OUTLIER_ELIMINATION = False

PATCH_ADAPTATION_MODE = "resize"

"""FILTER_KERNEL_V = np.array([[ 1, 2, 2, 2, 1],
                            [ 0, 0, 0, 0, 0],
                            [-1,-2,-2,-2,-1]]) / 8.0"""
"""FILTER_KERNEL_V = np.array([[ 1, 1, 1, 1, 1],
                            [ 0, 0, 0, 0, 0],
                            [-1,-1,-1,-1,-1]]) / 5.0"""
FILTER_KERNEL_H = np.array([[1, 1,  1, 1, 1],
                            [ 0, 0, 0, 0, 0],
                            [-1,-1,-1,-1,-1]]) / 5.0
FILTER_KERNEL_V = FILTER_KERNEL_H.T


def plot(img):
    plt.figure(figsize=(28, 28))
    #plt.axis("off")
    plt.imshow(img.astype("uint8"))
    plt.show()


def process_image(img, n_center_points=6):
    orig_img = img

    binarized_image = hinted_image(img)

    img = np.where(
        hinted_image(img, keepdims=True),
        np.ones(img.shape[:2] + (1,)) * np.array([[(255., 0., 255.)]]),
        img,
    )

    center_points = get_center_points(orig_img, n_v=1, n_h=n_center_points)

    # Sort center points
    center_points = sorted(center_points)

    for center_point in center_points:
        print("Mark", center_point)
        img = mark_point(center_point, img, color_val=(0, 0, 255), r=2)

    global debug_img
    debug_img = img

    map_components = get_map_components(orig_img, center_points, binarized_image=binarized_image)

    corrected_map = rebuild_map(map_components)

    return corrected_map.astype("uint8"), debug_img.astype("uint8")


def hinted_image(img, keepdims=False, border=0):
    hi = np.mean(img, axis=-1, keepdims=keepdims) < 170

    if border != 0:
        hi_new = np.zeros_like(hi)

        hi_new[border:-border, border:-border] = hi[border:-border, border:-border]

        hi = hi_new

    return hi


def vh_histograms(map):
    return np.sum(map, axis=1), np.sum(map, axis=0)


def get_extrema(type, hist, n, disallow_r=0):
    if type == "max":
        ranked_points = list(np.argsort(hist))
    elif type == "min":
        ranked_points = list(np.argsort(hist))[::-1]
    else:
        raise ValueError("Invalid type of extremum: ", type)

    points = []

    while len(points) < n and ranked_points:
        cp = ranked_points.pop()

        if len(points) == 0:
            points.append(cp)
        else:
            if all(abs(prev_point - cp) > disallow_r for prev_point in points):
                points.append(cp)

    return points


def get_center_points(img, n_v=1, n_h=1):
    hi = hinted_image(img, border=30)
    v_hist, h_hist = vh_histograms(hi)

    max_v = get_extrema("max", v_hist, n_v, disallow_r=10)
    max_h = get_extrema("max", h_hist, n_h, disallow_r=10)

    max_points = []

    for x in max_h:
        for y in max_v:
            max_points.append((int(y), int(x)))

    return max_points


def mark_point(point, img, color_val=(255, 255, 255), r=1):
    img = img.copy()

    img[int(point[0]) - r:int(point[0]) + r, int(point[1]) - r:int(point[1]) + r] = color_val

    return img


def get_point_of_intersection(l1, l2):
    # v * (x1, y1) + (nx1, ny1) = w * (x2, y2) + (nx2, ny2)
    lhs = np.array([l1[0], -l2[0]]).T
    rhs = l2[1].T - l1[1].T

    """print()
    print(l1)
    print(l2)

    print("LHS:", lhs)
    print("RHS:", rhs)"""

    solution = np.linalg.solve(lhs, rhs)

    factor = solution[0]

    intersection = l1[0] * factor + l1[1]

    #print("Intersection:", intersection)
    global debug_img
    debug_img = mark_point(intersection[::-1], debug_img, (255, 0, 0), r=2)

    return intersection


def estimate_edge_line_params(coords, vertical=False):
    regression = REGRESSOR()

    coords = list(coords)

    global debug_img
    for vy, vx in coords:
        debug_img = mark_point((vy, vx), debug_img, (0, 255, 0), r=2)

    y, x = zip(*coords)

    x = np.array(x).reshape((-1, 1))
    y = np.array(y).reshape((-1, 1))

    if vertical:
        x, y = y, x

    regression.fit(x, y)

    m = regression.coef_
    n = regression.intercept_

    m = np.ravel(m)[0]
    n = np.ravel(n)[0]

    # Convert into lines defined by vectors
    if not vertical:
        m = np.array((1, m))
        n = np.array((0, n))
    else:
        m = np.array((m, 1))
        n = np.array((n, 0))

    return m, n


def select_regression_points(area, indices_y, indices_x, n_considered_points, method="max"):
    if method == "max":
        nz_indices = np.nonzero(area)
        joint = np.array(list(zip(*nz_indices)))

        joint = sorted(joint, key=lambda idx: area[idx[0], idx[1]])

        y_coords, x_coords = zip(*joint)

        y_coords = y_coords[:n_considered_points]
        x_coords = x_coords[:n_considered_points]
    elif method == "mode_y" or method == "mode_x":
        nz_indices = np.nonzero(area)
        joint = np.transpose(nz_indices)
        joint = np.array(sorted(joint, reverse=True, key=lambda idx: area[idx[0], idx[1]])[:n_considered_points])

        if method == "mode_y":
            m = mode(joint[:, 0])[0][0]
            print(method, m)
            joint = np.array([v for v in joint if v[0] == m])
        else:
            m = mode(joint[:, 1])[0][0]
            joint = np.array([v for v in joint if v[1] == m])

        y_coords, x_coords = zip(*joint)

        y_coords = y_coords#[:n_considered_points]
        x_coords = x_coords#[:n_considered_points]
    elif method in ("topmost", "bottommost", "leftmost", "rightmost"):
        nz_indices = np.nonzero(area)

        joint = np.array(list(zip(*nz_indices)))

        joint = sorted(joint, reverse=True, key=lambda idx: area[idx[0], idx[1]])[:3 * n_considered_points]

        if OUTLIER_ELIMINATION:
            print("Len joint before", len(joint))

            is_inlier = LocalOutlierFactor(20).fit_predict(joint)

            joint = np.array([v for v, inlier in zip(joint, is_inlier) if inlier == 1])

            print("Len joint after", len(joint))

        if method == "topmost":
            ranked = np.array(sorted(joint, key=lambda v: v[0]))
        elif method == "bottommost":
            ranked = np.array(sorted(joint, key=lambda v: v[0], reverse=True))
        elif method == "leftmost":
            ranked = np.array(sorted(joint, key=lambda v: v[1]))
        elif method == "rightmost":
            ranked = np.array(sorted(joint, key=lambda v: v[1], reverse=True))

        y_coords, x_coords = zip(*ranked)

        y_coords = y_coords[:n_considered_points]
        x_coords = x_coords[:n_considered_points]
    else:
        raise ValueError("Invalid method: " + method)

    return y_coords, x_coords


def correct_center_point(center_point, image, window_width, window_height):
    # Create binary masks for each window quadrant
    indices_y = np.repeat(np.arange(image.shape[0])[:, None], image.shape[1], axis=1)
    indices_x = np.repeat(np.arange(image.shape[1])[None, :], image.shape[0], axis=0)

    cond = np.logical_and(
        np.logical_and(
            np.less_equal(
                center_point[0] - window_width,
                indices_y
            ),
            np.less(
                indices_y,
                center_point[0] + window_width,
            )
        ),
        np.logical_and(
            np.less_equal(
                center_point[1] - window_height,
                indices_x
            ),
            np.less(
                indices_x,
                center_point[1] + window_height
            )
        ),
    )

    map_window = np.where(
        cond[:, :, None],
        image,
        np.ones_like(image) * np.array([[[255, 255, 255]]])
    )

    center_point = get_center_points(map_window, n_v=1, n_h=1)[0]

    return center_point


def get_window_line_coords(vertical_edge_map, horizontal_edge_map, center_point, window_width, window_height,
                           search_width=8, n_considered_points=20):
    # Create binary masks for each window quadrant
    indices_y = np.repeat(np.arange(vertical_edge_map.shape[0])[:, None], vertical_edge_map.shape[1], axis=1)
    indices_x = np.repeat(np.arange(vertical_edge_map.shape[1])[None, :], vertical_edge_map.shape[0], axis=0)

    dirs = [(-1, -1), (-1, 1), (1, 1), (1, -1)]

    vmaps = []
    hmaps = []

    print("Center point:", center_point)

    tolerance = 2

    for dy, dx in dirs:
        top_vmap = min(center_point[0] + dy * (window_height // 2), center_point[0])
        bottom_vmap = max(center_point[0] + dy * (window_height // 2), center_point[0])
        left_vmap = min(center_point[1] + dx * search_width, center_point[1] - dx * tolerance)
        right_vmap = max(center_point[1] + dx * search_width, center_point[1] - dx * tolerance)

        top_hmap = min(center_point[0] + dy * search_width, center_point[0] - dy * tolerance)
        bottom_hmap = max(center_point[0] + dy * search_width, center_point[0] - dy * tolerance)
        left_hmap = min(center_point[1] + dx * (window_width // 2), center_point[1])
        right_hmap = max(center_point[1] + dx * (window_width // 2), center_point[1])

        cond_vmap = np.logical_and(
            np.logical_and(
                np.less_equal(
                    top_vmap,
                    indices_y
                ),
                np.less(
                    indices_y,
                    bottom_vmap
                )
            ),
            np.logical_and(
                np.less_equal(
                    left_vmap,
                    indices_x
                ),
                np.less(
                    indices_x,
                    right_vmap
                )
            ),
        )

        cond_hmap = np.logical_and(
            np.logical_and(
                np.less_equal(
                    top_hmap,
                    indices_y
                ),
                np.less(
                    indices_y,
                    bottom_hmap
                )
            ),
            np.logical_and(
                np.less_equal(
                    left_hmap,
                    indices_x
                ),
                np.less(
                    indices_x,
                    right_hmap
                )
            ),
        )

        # Mask out the irrelevant areas
        vmaps.append(np.where(
            cond_vmap,
            vertical_edge_map,
            np.zeros_like(vertical_edge_map),
        ))
        # Mask out the irrelevant areas
        hmaps.append(np.where(
            cond_hmap,
            horizontal_edge_map,
            np.zeros_like(horizontal_edge_map),
        ))

        #print(top, left, bottom, right)
        """print("Vertical:")
        plot(np.abs(vmaps[-1]))
        print("Horizontal:")
        plot(np.abs(hmaps[-1]))"""

    # Order the values to gather line regression points from in clock-wise fashion, starting with the top left line
    line_areas = [vmaps[0], vmaps[1], hmaps[1], hmaps[2], vmaps[2], vmaps[3], hmaps[3], hmaps[0]]
    map_multipliers = [-1, 1, -1, 1, 1, -1, 1, -1]
    #methods = ["leftmost", "rightmost", "topmost", "bottommost", "rightmost", "leftmost", "bottommost", "topmost"]
    #methods = ["max"] * 8
    methods = ["mode_x", "mode_x", "mode_y", "mode_y", "mode_x", "mode_x", "mode_y", "mode_y"]
    vertical_regression = [True, True, False, False, True, True, False, False]

    line_coords = []

    # Now, get line coords by determining the maximum points of the (absolute) edge values and calculating a regression
    # line for a subset of these points
    for map_multiplier, vertical, method, line_area in zip(map_multipliers, vertical_regression, methods, line_areas):
        y_coords, x_coords = select_regression_points(
            line_area * map_multiplier,
            indices_y,
            indices_x,
            n_considered_points,
            method=method
        )

        line_coords.append(estimate_edge_line_params(zip(y_coords, x_coords), vertical=vertical))

    return line_coords


def get_corners_ltrb(corners):
    """ Get the left, top, right and bottom dimensions of the corners """
    l = min(corners[0][0], corners[3][0])
    t = min(corners[0][1], corners[1][1])
    r = max(corners[1][0], corners[2][0])
    b = max(corners[2][1], corners[3][1])

    return l, t, r, b


def align_corners(corners):
    l, t, r, b = get_corners_ltrb(corners)

    corners = [
        np.array((l, t)),
        np.array((r, t)),
        np.array((r, b)),
        np.array((l, b)),
    ]

    return corners


def get_map_components(img, center_points, binarized_image, n_rows=2):
    #vertical_edge_map = filters.sobel_v(np.mean(img, axis=-1))
    #horizontal_edge_map = filters.sobel_h(np.mean(img, axis=-1))
    #vertical_edge_map = filters.sobel_v(binarized_image)
    #horizontal_edge_map = filters.sobel_h(binarized_image)
    #vertical_edge_map = filters.prewitt_v(binarized_image)
    #horizontal_edge_map = filters.prewitt_h(binarized_image)
    #vertical_edge_map = filters.prewitt_v(np.mean(img, axis=-1))
    #horizontal_edge_map = filters.prewitt_h(np.mean(img, axis=-1))
    vertical_edge_map = filters.edges.convolve(np.mean(img, axis=-1), FILTER_KERNEL_V)
    horizontal_edge_map = filters.edges.convolve(np.mean(img, axis=-1), FILTER_KERNEL_H)

    #vertical_edge_map = np.mean([filters.edges.convolve(img[:, :, i], FILTER_KERNEL_V) for i in range(3)], axis=0)
    #horizontal_edge_map = np.mean([filters.edges.convolve(img[:, :, i], FILTER_KERNEL_H) for i in range(3)], axis=0)
    #vertical_edge_map2 = filters.edges.convolve(np.mean(img, axis=-1), FILTER_KERNEL_V[::-1, ::-1])
    #horizontal_edge_map2 = filters.edges.convolve(np.mean(img, axis=-1), FILTER_KERNEL_H[::-1, ::-1])
    #vertical_edge_map = filters.edges.convolve(binarized_image, FILTER_KERNEL_V)
    #horizontal_edge_map = filters.edges.convolve(binarized_image, FILTER_KERNEL_H)

    #vertical_edge_map = np.maximum(vertical_edge_map, vertical_edge_map2)
    #horizontal_edge_map = np.maximum(horizontal_edge_map, horizontal_edge_map2)

    print("Edge maps:", vertical_edge_map.shape, horizontal_edge_map.shape, img.shape)

    plot(vertical_edge_map)
    plot(horizontal_edge_map)

    # Arrange center points in a grid
    center_points_grid = np.array(center_points).reshape((n_rows - 1, -1, 2))

    window_width = int(2 * 0.9 * img.shape[1] / (center_points_grid.shape[1] + 1))
    window_height = int(2 * 0.9 * img.shape[0] / n_rows)
    #window_width, window_height = 50, 200

    print("Window width, height: ", window_width, window_height)

    component_lines_horizontal = [[(np.array((1, 0)), np.array((0, 0))) for _ in range(len(center_points_grid[0]) + 1)]]
    component_lines_vertical = []

    for r_id, center_points_row in enumerate(center_points_grid):
        component_lines_horizontal_row_top = []
        component_lines_horizontal_row_bottom = []
        component_lines_vertical_row = []
        component_lines_vertical_row_last = []

        # Add initial left border
        component_lines_vertical_row.append((np.array((0, 1)), np.array((0, 0))))
        component_lines_vertical_row_last.append((np.array((0, 1)), np.array((0, 0))))

        for c_id, center_point in enumerate(center_points_row):
            corrected_center_point = correct_center_point(
                center_point,
                img,
                50,
                50,
            )

            print(center_point, "corrected to", corrected_center_point)
            global debug_img
            debug_img = mark_point(corrected_center_point, debug_img, color_val=(255, 170, 100), r=5)

            wlc = get_window_line_coords(
                vertical_edge_map=vertical_edge_map,
                horizontal_edge_map=horizontal_edge_map,
                center_point=corrected_center_point,
                window_width=window_width, # TODO: Make values relative to center points
                window_height=window_height,
                search_width=SEARCH_WIDTH,
                n_considered_points=N_CONSIDERED_POINTS,
            )

            # Add right border, then the next left border
            component_lines_vertical_row.append(wlc[0])
            component_lines_vertical_row.append(wlc[1])

            component_lines_vertical_row_last.append(wlc[5])
            component_lines_vertical_row_last.append(wlc[4])

            # Add the lines left of the center point
            component_lines_horizontal_row_top.append(wlc[7])
            component_lines_horizontal_row_bottom.append(wlc[6])

        # For the last component we need to take the right lines of the last center point, as there are
        # no following center points of which we could take the left ones
        component_lines_horizontal_row_top.append(wlc[2])
        component_lines_horizontal_row_bottom.append(wlc[3])

        component_lines_vertical_row.append((np.array((0, 1)), np.array((img.shape[1] - 1, 0))))
        component_lines_vertical_row_last.append((np.array((0, 1)), np.array((img.shape[1] - 1, 0))))

        component_lines_horizontal.append(component_lines_horizontal_row_top)
        component_lines_horizontal.append(component_lines_horizontal_row_bottom)
        component_lines_vertical.append(component_lines_vertical_row)

    # For the vertical lines of the last row we again use the bottom lines from the previous last centerpoints
    component_lines_vertical.append(component_lines_vertical_row_last)

    # Add the concluding horizontal lines
    component_lines_horizontal.append([(np.array((1, 0)), np.array((0, img.shape[0] - 1))) for _ in range(len(center_points_grid[0]) + 1)])

    print("Component lines")
    print(component_lines_horizontal)
    print(component_lines_vertical)

    # Now we have all border lines, time to bring them together for each component, calculate the corner points and
    # collect the component areas
    components = []

    for row_id, component_lines_vertical_row in enumerate(component_lines_vertical):
        components_row = []
        for component_id in range(len(component_lines_horizontal[row_id])):
            # Calculate the four corner points of every center point, starting top left, clockwise direction
            corners = [
                get_point_of_intersection( # Top left corner
                    component_lines_vertical_row[2 * component_id],  # Left
                    component_lines_horizontal[2 * row_id][component_id],  # Top
                ),
                get_point_of_intersection( # Top right corner
                    component_lines_horizontal[2 * row_id][component_id],  # Top
                    component_lines_vertical_row[2 * component_id + 1],  # Right
                ),
                get_point_of_intersection( # Bottom right corner
                    component_lines_vertical_row[2 * component_id + 1],  # Right
                    component_lines_horizontal[2 * row_id + 1][component_id],  # Bottom
                ),
                get_point_of_intersection( # Bottom left corner
                    component_lines_horizontal[2 * row_id + 1][component_id],  # Bottom
                    component_lines_vertical_row[2 * component_id],  # Left
                )
            ]

            # For now, align corners according to their maximum width and height in any direction
            # TODO: Alternatively: we could rotate the components until they are completely orthogonal
            corners = align_corners(corners)
            l, t, r, b = get_corners_ltrb(corners)

            l, t, r, b = int(l), int(t), int(r), int(b)

            print(l, t, r, b)

            # Cut out the relevant part of the map, leave out the top and left lines --> start one pixel later
            component = img[t + 1:b, l + 1:r]

            components_row.append(component)

        components.append(components_row)

    return components


def rebuild_map(components):
    col_widths = [0] * len(components[0])
    row_heights = [0] * len(components)

    print("Map")

    # Find the same (maximum sizes) for each component
    for r_id, component_row in enumerate(components):
        for c_id, component in enumerate(component_row):
            print(component.shape)
            h, w, d = component.shape

            if h > row_heights[r_id]:
                row_heights[r_id] = h
            if w > col_widths[c_id]:
                col_widths[c_id] = w

    print(row_heights)
    print(sum(row_heights))

    full_map = np.zeros((sum(row_heights), sum(col_widths), 3))

    for r_id, component_row in enumerate(components):
        for c_id, component in enumerate(component_row):
            h, w, d = component.shape
            target_w, target_h = col_widths[c_id], row_heights[r_id]

            if PATCH_ADAPTATION_MODE == "pad":
                # Bring all map components to the same size by padding them (TODO: Try bilinear resize)
                pad_left = math.floor((target_w - w) / 2)
                pad_right = math.ceil((target_w - w) / 2)
                pad_top = math.floor((target_h - h) / 2)
                pad_bottom = math.ceil((target_h - h) / 2)

                component = np.pad(
                    component,
                    (
                        (pad_top, pad_bottom),
                        (pad_left, pad_right),
                        (0, 0),
                    ),
                    mode="constant",
                )
            elif PATCH_ADAPTATION_MODE == "resize":
                img_pil = Image.fromarray(component)
                component = np.array(img_pil.resize((target_w, target_h)))
            else:
                raise ValueError("Invalid adaptation mode: " + PATCH_ADAPTATION_MODE)

            map_coord_y = sum(row_heights[:r_id])
            map_coord_x = sum(col_widths[:c_id])

            full_map[
                map_coord_y:map_coord_y + target_h,
                map_coord_x:map_coord_x + target_w,
            ] = component

    return full_map