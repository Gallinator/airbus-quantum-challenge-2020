from pathlib import Path
import cairosvg
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from aircraft_data import AircraftData
from loading_problem import LoadingProblem

CONT_BOX_COLOR = {'t1': '#cdeda3', 't2': '#dce7c8', 't3': '#bcece7'}
TEXT_COLOR = '#151e0b'
IMG_SIZE = (3840, 2160)


def plot_solution(title: str, problem: LoadingProblem, cont_occ: np.ndarray, out_path: str):
    Path('out').mkdir(exist_ok=True)
    cairosvg.svg2png(url='resources/aircraft_bg_res.svg', write_to=out_path, output_width=IMG_SIZE[0],
                     output_height=IMG_SIZE[1])
    bg_img = Image.open(out_path)
    img = Image.new(mode='RGBA', size=IMG_SIZE, color='#FFFFFF00')
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", int(img.size[1] * 0.05))
    w, h = img.size
    margin_x = (w * 0.143, w * 0.195)
    margin_y = h * 0.05
    draw_area_size = (w - (margin_x[0] + margin_x[1]), h - margin_y * 2)

    draw_containers(img, cont_occ, problem, 20.0,
                    draw_area_size=draw_area_size,
                    draw_area_start_x=margin_x[0],
                    font=font)

    fig, axs = plt.subplots(3, figsize=(7, 8), sharex=False, height_ratios=[4, 1, 2])
    fig.suptitle(title)

    axs[0].imshow(bg_img)
    axs[0].imshow(img)
    add_legend(axs[0])
    axs[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    image_margin = IMG_SIZE[0] * 0.1
    axs[0].set_xlim(-image_margin, IMG_SIZE[0] + image_margin)

    plot_cg(axs[1], problem, cont_occ)
    plot_shear(axs[2], problem, cont_occ)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.show()


def plot_shear(ax: Axes, problem: LoadingProblem, cont_occ: np.ndarray):
    shear_l, shear_r = problem.get_shear(cont_occ)

    # Fill the left end of the curves
    y = np.insert(problem.aircraft.shear_curve, 0, 0)
    shear_l = np.insert(shear_l, 0, 0)

    x_l = problem.aircraft.location_ends[problem.aircraft.location_ends <= 0]
    x_l = np.insert(x_l, 0, -problem.aircraft.payload_area_length / 2)
    x_r = problem.aircraft.location_ends[problem.aircraft.location_ends >= 0]

    # Add center location end if odd positions
    if problem.aircraft.num_positions % 2 != 0:
        x_l = np.concatenate([x_l, [0]])
        x_r = np.concatenate([[0], x_r])

    ax.plot(x_l, y[:len(x_l)], c='red', label='Max shear')
    ax.plot(x_r, y[len(x_l):], c='red')
    ax.plot(x_l, shear_l, c='green', label='Actual shear')
    ax.plot(x_r, shear_r, c='green')
    ax.set_xlabel('x')
    ax.set_ylabel('Shear [N]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.legend()


def plot_cg(ax: Axes, problem: LoadingProblem, cont_occ: np.ndarray):
    lim = problem.aircraft.payload_area_length / 2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, 2)
    ax.tick_params(left=False, labelleft=False)
    cg = problem.get_cg(cont_occ)
    error = [[abs(problem.aircraft.min_cg)], [abs(problem.aircraft.max_cg)]]
    ax.set_xlabel('x')
    ax.errorbar(0, 1, xerr=error, capsize=10)
    ax.scatter(problem.target_cg, 1, marker='x', c='black', label='Target CG')
    ax.scatter(cg, 1, marker='o', c='green', label='Actual CG')
    ax.legend()


def add_legend(ax: Axes):
    ax.legend(handles=[
        mpatches.Patch(color=CONT_BOX_COLOR['t1'], label='Type 1'),
        mpatches.Patch(color=CONT_BOX_COLOR['t2'], label='Type 2'),
        mpatches.Patch(color=CONT_BOX_COLOR['t3'], label='Type 3'),
    ], loc='upper left')


def get_container_draw_area_size(aircraft: AircraftData, draw_area_size: tuple) -> tuple:
    return draw_area_size[0] / aircraft.num_positions, min(300, draw_area_size[1])


def draw_containers(img: Image, cont_occ: np.ndarray, problem: LoadingProblem, cont_margin_x, draw_area_start_x,
                    draw_area_size, font):
    draw = ImageDraw.Draw(img)
    cont_area_w, cont_area_h = get_container_draw_area_size(problem.aircraft, draw_area_size)

    img_center_y = img.size[1] / 2
    # This will be updated every time a whole box is drawn
    next_draw_start_x = draw_area_start_x
    pos = 0
    while pos < problem.aircraft.num_positions:
        pos_increase = 1
        # Get row as list
        pos_data = cont_occ[:, pos].flatten()
        cont_indices = np.where(pos_data == 1)[0]

        # Calculate container drawing area bounding box
        draw_start = next_draw_start_x
        draw_bottom = img_center_y + cont_area_h / 2.0
        draw_top = img_center_y - cont_area_h / 2.0
        draw_end = draw_start + cont_area_w

        for i, c in enumerate(cont_indices):
            cont_start = draw_start + cont_margin_x
            cont_label = str(problem.container_masses[c])
            cont_color = CONT_BOX_COLOR[problem.container_types[c]]
            cont_end = cont_start + cont_area_w - 2 * cont_margin_x

            match problem.container_types[c]:
                case 't1':
                    left_top, right_bottom = (cont_start, draw_top), (cont_end, draw_bottom)
                    draw.rounded_rectangle([left_top, right_bottom], fill=cont_color)
                    draw_text(draw, font, cont_label, left_top, right_bottom)
                # Both containers in the same position are drawn
                case 't2':
                    cont_top = draw_top + i * cont_area_h / 2
                    cont_bottom = cont_top + cont_area_h / 2 - cont_margin_x
                    left_top, right_bottom = (cont_start, cont_top), (cont_end, cont_bottom)
                    draw.rounded_rectangle([left_top, right_bottom], fill=cont_color)
                    draw_text(draw, font, cont_label, left_top, right_bottom)
                # Assume contiguous containers
                case 't3':
                    cont_end = cont_start + 2 * cont_area_w - 2 * cont_margin_x
                    left_top, right_bottom = (cont_start, draw_top), (cont_end, draw_bottom)
                    draw.rounded_rectangle([left_top, right_bottom], fill=cont_color)
                    draw_text(draw, font, cont_label, left_top, right_bottom)
                    draw_end = cont_end + cont_margin_x
                    pos_increase = 2

        next_draw_start_x = draw_end
        pos += pos_increase


def draw_text(draw: ImageDraw.Draw, font, text: str, cont_left_top: tuple, cont_right_bottom: tuple):
    txt_pos = ((cont_left_top[0] + cont_right_bottom[0]) / 2, (cont_left_top[1] + cont_right_bottom[1]) / 2)
    draw.text(txt_pos, text, font=font, fill=TEXT_COLOR, anchor='mm')
