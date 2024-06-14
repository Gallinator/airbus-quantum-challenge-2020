from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from aircraft_data import AircraftData
from loading_problem import LoadingProblem

CONT_BOX_COLOR = {'t1': '#cdeda3', 't2': '#dce7c8', 't3': '#bcece7'}
TEXT_COLOR = '#151e0b'


def plot_solution(problem: LoadingProblem, cont_occ: np.ndarray):
    img = Image.new(mode='RGB', size=(3840, 2160), color=(255, 255, 255))
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", int(img.size[1] * 0.05))
    w, h = img.size
    margin_x, margin_y = (w * 0.02, h * 0.05)
    draw_area_size = (w - margin_x * 2, h - margin_y * 2)

    draw_containers(img, cont_occ, problem, 20.0, draw_area_size, font)

    plt.imshow(img)
    plt.show()


def get_container_draw_area_size(aircraft: AircraftData, draw_area_size: tuple) -> tuple:
    return draw_area_size[0] / aircraft.num_positions, min(512, draw_area_size[1])


def draw_containers(img: Image, cont_occ: np.ndarray, problem: LoadingProblem, cont_margin_x, draw_area_size, font):
    draw = ImageDraw.Draw(img)
    cont_area_w, cont_area_h = get_container_draw_area_size(problem.aircraft, draw_area_size)

    img_center_y = img.size[1] / 2
    # This will be updated every time a whole box is drawn
    next_draw_start_x = (img.size[0] - draw_area_size[0]) / 2
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

            match problem.container_types[c]:
                case 't1':
                    cont_end = cont_start + cont_area_w - 2 * cont_margin_x
                    draw.rounded_rectangle([(cont_start, draw_top), (cont_end, draw_bottom)], fill=cont_color)
                    draw_text(draw, font, cont_label, cont_start, cont_end, img_center_y)
                # Both containers in the same position are drawn
                case 't2':
                    cont_start = draw_start + i * (cont_area_w / 2) + cont_margin_x
                    cont_end = cont_start + cont_area_w / 2 - 2 * cont_margin_x
                    draw.rounded_rectangle([(cont_start, draw_top), (cont_end, draw_bottom)], fill=cont_color)
                    draw_text(draw, font, cont_label, cont_start, cont_end, img_center_y)
                # Assume contiguous containers
                case 't3':
                    cont_end = cont_start + 2 * cont_area_w - 2 * cont_margin_x
                    draw.rounded_rectangle([(cont_start, draw_top), (cont_end, draw_bottom)], fill=cont_color)
                    draw_text(draw, font, cont_label, cont_start, cont_end, img_center_y)
                    pos_increase = 2

        next_draw_start_x = draw_end
        pos += pos_increase


def draw_text(draw: ImageDraw.Draw, font, text: str, cont_start, cont_end, center_y):
    txt_pos = ((cont_start + cont_end) / 2, center_y)
    draw.text(txt_pos, text, font=font, fill=TEXT_COLOR, anchor='mm')
