import os
import numpy as np
import torch
from tqdm import trange
import netCDF4 as nc
import cv2
import matplotlib
import argparse
from omegaconf import OmegaConf
import ood_mahalanobis
import matplotlib.pyplot as plt
from PIL import Image

config = None
name = None
dev = 'cpu'
args = None
compounds_numbers = None
numbers_compounds = {}

system_number = 0
y_six_seconds = 0
x_six_seconds = 0
y_eight_seconds = 0
x_eight_seconds = 0
y_ten_seconds = 0
x_ten_seconds = 0
t1_shift = 0

triangles = []


def set_system_1(scan_length: int):
    '''
    Set parameters for system 1
    :param scan_length:
    :return: None
    '''
    global y_six_seconds, x_six_seconds, y_eight_seconds, x_eight_seconds, y_ten_seconds, x_ten_seconds
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 200
    # y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    y_eight_seconds = 1200
    x_eight_seconds = 149
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds


def set_system_2(scan_length: int):
    '''
    Set parameters for system3
    :param scan_length: length of scan
    :return: None
    '''
    global y_six_seconds, x_six_seconds, y_eight_seconds, x_eight_seconds, y_ten_seconds, x_ten_seconds
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 200
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds


def set_system_3(scan_length: int):
    '''
    Set parameters for system3
    :param scan_length: length of scan
    :return: None
    '''
    global y_six_seconds, x_six_seconds, y_eight_seconds, x_eight_seconds, y_ten_seconds, x_ten_seconds
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 200
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds


def fill_spectrogram_image_torch(ds, spectrogram_image, data_pointer, x_start, x_length, y_length,
                                 mass_range_min, mass_max_range_len):
    '''
    Fill spectrogram image with data from cdf file
    :param ds: loaded cdf file
    :param spectrogram_image: 3D tensor which will be filled with data
    :param data_pointer: pointer to data in cdf file
    :param x_start: x coordinate where to start filling
    :param x_length: length of x coordinate
    :param y_length: length of y coordinate
    :param mass_range_min: minimal mass value
    :param mass_max_range_len: length of mass range
    :return: updated data_pointer
    '''
    for j in trange(x_start, x_start + x_length):
        for i in range(0, y_length):
            si_s = ds['scan_index'][data_pointer]
            si_t = ds['scan_index'][data_pointer] + ds['point_count'][data_pointer]
            indexes = (ds['mass_values'][si_s:si_t] - mass_range_min).to(torch.long)
            mass_data = torch.zeros((mass_max_range_len,), dtype=torch.int32).to(dev)
            mass_data[indexes] = ds['intensity_values'][si_s:si_t]
            spectrogram_image[:, i, j] = mass_data
            data_pointer += 1
    return data_pointer


def load_cdf(cdf_path: str):
    '''
    Load cdf file and return spectrogram image
    :param cdf_path: path to cdf file which will be loaded
    :return: spectrogram image - 3D tensor
            ds - loaded cdf file
    '''
    global config, dev, x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds
    saved_file = os.path.join(os.getcwd(), 'tmp', cdf_path.split('/')[-1] + '_' + str(system_number) + '.pth')
    saved_file_raw = os.path.join(os.getcwd(), 'tmp', cdf_path.split('/')[-1] + '_' + str(system_number) + '_raw.pth')
    if (os.path.exists(saved_file)):
        spectrogram_image = torch.load(saved_file)
        ds = torch.load(saved_file_raw)
        if (config.constants.system_number == 1):
            set_system_1(len(ds['total_intensity']))
        elif (config.constants.system_number == 2):
            set_system_2(len(ds['total_intensity']))
        elif (config.constants.system_number == 3):
            set_system_3(len(ds['total_intensity']))
        else:
            raise ValueError('Unknown system number')
    else:
        ds = nc.Dataset(cdf_path)
        data = {
            'mass_range_min': torch.tensor(ds['mass_range_min'][:], device=dev),
            'scan_index': torch.tensor(ds['scan_index'][:], device=dev),
            'point_count': torch.tensor(ds['point_count'][:], device=dev),
            'mass_values': torch.tensor(ds['mass_values'][:], device=dev),
            'intensity_values': torch.tensor(ds['intensity_values'][:], device=dev),
            'total_intensity': torch.tensor(ds['total_intensity'][:], device=dev),
            'mass_range_max': torch.tensor(ds['mass_range_max'][:], device=dev)
        }
        ds = data
        if (config.constants.system_number == 1):
            set_system_1(len(ds['total_intensity']))
        elif (config.constants.system_number == 2):
            set_system_2(len(ds['total_intensity']))
        elif (config.constants.system_number == 3):
            set_system_3(len(ds['total_intensity']))
        else:
            raise ValueError('Unknown system number')
        mass_range_min = 0
        mass_max_range_len = int(ds['mass_range_max'][:].max()) + 1
        spectrogram_image = torch.zeros((mass_max_range_len, y_ten_seconds,
                                         x_six_seconds + x_eight_seconds + x_ten_seconds),
                                        dtype=torch.int32).to(dev)
        data_pointer_spectrogram = fill_spectrogram_image_torch(ds, spectrogram_image,
                                                                0,
                                                                0, x_six_seconds, y_six_seconds,
                                                                mass_range_min, mass_max_range_len)
        data_pointer_spectrogram = fill_spectrogram_image_torch(ds, spectrogram_image,
                                                                data_pointer_spectrogram,
                                                                x_six_seconds,
                                                                x_eight_seconds, y_eight_seconds,
                                                                mass_range_min,
                                                                mass_max_range_len)
        _ = fill_spectrogram_image_torch(ds, spectrogram_image,
                                         data_pointer_spectrogram,
                                         x_six_seconds + x_eight_seconds, x_ten_seconds, y_ten_seconds,
                                         mass_range_min, mass_max_range_len)
        cur_dir = os.getcwd()
        cur_dir = cur_dir + '/tmp'
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        torch.save(spectrogram_image, saved_file)
        torch.save(ds, saved_file_raw)
    return spectrogram_image, ds


def load_ref_compounds():
    '''
    Load compounds and their signatures from yaml file
    :return: dictionary with compounds and their signatures
    '''
    global system_number
    if (system_number == 1):
        return OmegaConf.load('compounds_system1.yaml')
    elif (system_number == 2):
        return OmegaConf.load('compounds_system2.yaml')
    elif (system_number == 3):
        return OmegaConf.load('compounds_system3.yaml')
    else:
        raise ValueError('Unknown system number')


def from_pixels_times(t1_pixel, t2_pixel):
    '''
    Convert pixels to times
    :param t1_pixel: pixel for t1
    :param t2_pixel: pixel for t2
    :return: tuple with times (t1, t2)
    '''
    t2_time = t2_pixel * 5
    t2_time = t2_time / 1000
    t2_time = float(t2_time)
    if system_number == 1:
        if t1_pixel < (x_six_seconds + x_eight_seconds):
            t1_time = t1_pixel * 6
        else:
            t1_ten_pixels = t1_pixel - (x_six_seconds + x_eight_seconds)
            t1_time = (x_six_seconds + x_eight_seconds) * 6 + t1_ten_pixels * 10
        t1_time -= t1_shift
    else:
        if t1_pixel < (x_six_seconds):
            t1_time = t1_pixel * 6
        elif t1_pixel < (x_six_seconds + x_eight_seconds):
            t1_time = (x_six_seconds) * 6 + (t1_pixel - x_six_seconds) * 8
        else:
            t1_ten_pixels = t1_pixel - (x_six_seconds + x_eight_seconds)
            t1_time = x_six_seconds * 6 + x_eight_seconds * 8 + t1_ten_pixels * 10
        t1_time -= t1_shift
    t1_time = int(t1_time)
    return (t1_time, t2_time)


def from_times_pixels(compounds: dict):
    '''
    Convert dictionary of compounds with times to pixels
    :param compounds:  dictionary with compounds and their times
    :return: dictionary with compounds and their pixels
    '''
    global x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, y_ten_seconds, x_ten_seconds, system_number, t1_shift
    compounds_pixels = {}
    if system_number == 1:
        # MINIMAL TIME WHEN THE MACHINE IS SET FOR 10 SECONDS IN FIRST COLUMN
        min_ten_seconds = x_six_seconds * 6 + x_eight_seconds * 6
        for compound in compounds:
            # T1 READ FROM FILE
            t1 = compounds[compound]['t1'] + t1_shift
            # T2 READ FROM FILE
            t2 = compounds[compound]['t2']

            # CONVERT TO MILLISECONDS
            t2 *= 1000
            # CALCULATE PIXELS FOR T2 - 5 MILLISECONDS PER PIXEL
            t2_pixels = t2 // 5

            # CHECK IF T1 IS LESS THAN MINIMAL TIME FOR 10 SECONDS
            if t1 < min_ten_seconds:
                # 6 SECONDS FOR ONE PIXEL
                t1_pixels = t1 // 6
            else:
                # SUBTRACT MINIMAL TIME FOR 10 SECONDS  TO GET TIME FOR DIVIDING BY 10
                t1 = t1 - min_ten_seconds
                # CALCULATE PIXELS FOR T1 - 10 SECONDS FOR ONE PIXEL
                # CALCULATE PIXELS FOR T1 - 10 SECONDS FOR ONE PIXEL
                t1_pixels = (x_six_seconds + x_eight_seconds) + t1 // 10
                # t1_pixels = (x_six_seconds + x_eight_seconds) + math.ceil(t1 / 10) - 1
            # SAVE PIXELS FOR COMPOUND
            t2_pixels = int(t2_pixels)
            t1_pixels = int(t1_pixels)
            compounds_pixels[compound] = (t1_pixels, t2_pixels)
    else:
        # MINIMAL TIME WHEN THE MACHINE IS SET FOR 10 SECONDS IN FIRST COLUMN
        min_ten_seconds = x_six_seconds * 6 + x_eight_seconds * 8
        min_eight_seconds = x_six_seconds * 6
        for compound in compounds:
            # T1 READ FROM FILE
            t1 = compounds[compound]['t1'] + t1_shift
            # T2 READ FROM FILE
            t2 = compounds[compound]['t2']

            # CONVERT TO MILLISECONDS
            t2 *= 1000
            # CALCULATE PIXELS FOR T2 - 5 MILLISECONDS PER PIXEL
            t2_pixels = t2 // 5

            # CHECK IF T1 IS LESS THAN MINIMAL TIME FOR 10 SECONDS
            if t1 < min_eight_seconds:
                # 6 SECONDS FOR ONE PIXEL
                t1_pixels = t1 // 6
            elif t1 < min_ten_seconds:
                t1 = t1 - min_eight_seconds
                t1_pixels = x_six_seconds + t1 // 8
            else:
                # SUBTRACT MINIMAL TIME FOR 10 SECONDS  TO GET TIME FOR DIVIDING BY 10
                t1 = t1 - min_ten_seconds
                # CALCULATE PIXELS FOR T1 - 10 SECONDS FOR ONE PIXEL
                # CALCULATE PIXELS FOR T1 - 10 SECONDS FOR ONE PIXEL
                t1_pixels = (x_six_seconds + x_eight_seconds) + t1 // 10
                # t1_pixels = (x_six_seconds + x_eight_seconds) + math.ceil(t1 / 10) - 1
            # SAVE PIXELS FOR COMPOUND
            t2_pixels = int(t2_pixels)
            t1_pixels = int(t1_pixels)
            compounds_pixels[compound] = (t1_pixels, t2_pixels)
    return compounds_pixels


def find_shift(compounds: dict, spectrogram_image: torch.Tensor):
    '''
    Find shift for each significant compound
    :param compounds: dictionary with compounds and their expected positions pixels
    :param spectrogram_image: spectrogram image (3D tensor)
    :return: avg_tup - average shift for all significant compounds
             ref_compounds_pixels - dictionary with reference pixels for each significant compound
             shifts_for_compounds - dictionary with shifts for each significant compound
    '''
    global config
    BOX_SIZE_ADD_X = config.constants['box_t1']
    BOX_SIZE_ADD_Y = config.constants['box_t2']
    avg_shift_x = 0
    avg_shift_y = 0
    counter = 0
    ref_compounds_pixels = {}
    shifts_for_compounds = {}
    for compound in config.calibration_compounds:
        spectrum = torch.load(f'avg_spectrum/{compound}.pth').to(dev)
        x_compound = compounds[compound][0]
        y_compound = compounds[compound][1]

        # normalize
        # cut to double
        new_x_compound, new_y_compound = find_position_dot(BOX_SIZE_ADD_X, BOX_SIZE_ADD_Y, spectrogram_image,
                                                           spectrum, x_compound, y_compound)
        if new_x_compound == -1 and new_y_compound == -1:
            continue
        ref_compounds_pixels[compound] = (new_x_compound, new_y_compound)
        shift_x = new_x_compound - x_compound
        shift_y = new_y_compound - y_compound
        shifts_for_compounds[compound] = (shift_x, shift_y)
        avg_shift_x += shift_x
        avg_shift_y += shift_y
        counter += 1
    avg_shift_x = avg_shift_x / counter
    avg_shift_y = avg_shift_y / counter
    avg_shift_x = int(avg_shift_x.item())
    avg_shift_y = int(avg_shift_y.item())
    avg_tup = (avg_shift_x, avg_shift_y)
    return avg_tup, ref_compounds_pixels, shifts_for_compounds


def find_position_dot(BOX_SIZE_ADD_X, BOX_SIZE_ADD_Y, spectrogram_image, spectrum, x_compound, y_compound):
    '''
    Find position in the box for compound using dot product
    :param BOX_SIZE_ADD_X: length to add to x coordinate in each direction
    :param BOX_SIZE_ADD_Y: length to add to y coordinate in each direction
    :param spectrogram_image: spectrogram image (3D tensor)
    :param spectrum: spectrum fof the compound
    :param x_compound: expected x coordinate of the compound
    :param y_compound: expected y coordinate of the compound
    :return:
    '''
    global config
    cut_spectrogram_image = spectrogram_image[:,
                            y_compound - BOX_SIZE_ADD_Y:y_compound + 1 + BOX_SIZE_ADD_Y,
                            x_compound - BOX_SIZE_ADD_X:x_compound + 1 + BOX_SIZE_ADD_X].clone()
    cut_spectrogram_image = cut_spectrogram_image.double()
    spectrum = spectrum.double()
    cut_spectrogram_image = torch.nn.functional.normalize(cut_spectrogram_image, dim=0)
    spectrum = torch.nn.functional.normalize(spectrum, dim=0)
    if cut_spectrogram_image.shape[0] > spectrum.shape[0]:
        cut_spectrogram_image = cut_spectrogram_image[:spectrum.shape[0], :, :]
    elif cut_spectrogram_image.shape[0] < spectrum.shape[0]:
        spectrum = spectrum[:cut_spectrogram_image.shape[0], :, :]
    dot_product = spectrum * cut_spectrogram_image
    # y, x
    dot_product = dot_product.sum(dim=0)
    index_of_max_dot_product = torch.argmax(dot_product)
    value_of_max_dot_product = dot_product.flatten()[index_of_max_dot_product]
    if (value_of_max_dot_product < config.constants['dot_product_threshold']):
        return -1, -1
    y_pos = index_of_max_dot_product // dot_product.shape[1]
    x_pos = index_of_max_dot_product % dot_product.shape[1]
    new_x_compound = x_compound - BOX_SIZE_ADD_X + x_pos
    new_y_compound = y_compound - BOX_SIZE_ADD_Y + y_pos
    return new_x_compound, new_y_compound


def barycentric_coordinates(point, triangle, pixels):
    '''
    Calculate barycentric coordinates
    :param point:  point for which we want to calculate barycentric coordinates
    :param triangle:  triangle for which we want to calculate barycentric coordinates
    :param pixels:  dictionary with pixels for each compound
    :return:  barycentric coordinates
    '''
    global numbers_compounds
    A = pixels[numbers_compounds[triangle[0]]]
    B = pixels[numbers_compounds[triangle[1]]]
    C = pixels[numbers_compounds[triangle[2]]]
    denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    if denominator == 0:
        return (-1, -1, -1)
    u = ((B[1] - C[1]) * (point[0] - C[0]) + (C[0] - B[0]) * (point[1] - C[1])) / denominator
    v = ((C[1] - A[1]) * (point[0] - C[0]) + (A[0] - C[0]) * (point[1] - C[1])) / denominator
    w = 1 - u - v
    if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:
        return (u, v, w)
    else:
        return (-1, -1, -1)


def find_positions(compounds_pixels: dict, compounds_all: dict, ref_compounds_pixels: dict, avg_shift: tuple,
                   spectrogram_image: torch.Tensor, shifts_for_compounds: dict = None):
    '''
    Find positions for each compound
    :param compounds_pixels: dictionary with compounds and their expected positions pixels
    :param compounds: dictionary with compounds and their specifications
    :param ref_compounds_pixels: dictionary with reference pixels for each significant compound
    :param avg_shift: average shift for all significant compounds
    :return: dictionary with positions for each compound and spectrum for each compound
    '''
    global args, triangles, numbers_compounds
    compounds_position = {}
    for compound in compounds_all:
        if (ref_compounds_pixels is not None) and (compound in ref_compounds_pixels):
            x_compound = ref_compounds_pixels[compound][0].item()
            y_compound = ref_compounds_pixels[compound][1].item()
        elif ref_compounds_pixels is None:
            raise ValueError('Reference compounds pixels are not set')
        else:
            x_compound = compounds_pixels[compound][0]
            y_compound = compounds_pixels[compound][1]
            if args.shift_method == 'triangles':
                found = False
                for triangle in triangles:
                    cord = barycentric_coordinates((x_compound, y_compound), triangle, ref_compounds_pixels)
                    if cord[0] != -1:
                        found = True
                        shift_x = int(
                            shifts_for_compounds[numbers_compounds[triangle[0]]][0] * cord[0] +
                            shifts_for_compounds[numbers_compounds[triangle[1]]][0] *
                            cord[1] + shifts_for_compounds[numbers_compounds[triangle[2]]][0] * cord[2])
                        shift_y = int(
                            shifts_for_compounds[numbers_compounds[triangle[0]]][1] * cord[0] +
                            shifts_for_compounds[numbers_compounds[triangle[1]]][1] *
                            cord[1] + shifts_for_compounds[numbers_compounds[triangle[2]]][1] * cord[2])
                        x_compound += shift_x
                        y_compound += shift_y
                    if not found:
                        x_compound += avg_shift[0]
                        y_compound += avg_shift[1]
            else:
                x_compound += avg_shift[0]
                y_compound += avg_shift[1]
        spectrum = torch.load(f'avg_spectrum/{compound}.pth').to(dev)
        x_compound, y_compound = find_position_dot(compounds_all[compound]['box_t1'], compounds_all[compound]['box_t2'],
                                                   spectrogram_image, spectrum, x_compound,
                                                   y_compound)
        if x_compound == -1 and y_compound == -1:
            continue
        x_compound = int(x_compound)
        y_compound = int(y_compound)
        spectrum_found = spectrogram_image[:, y_compound, x_compound].clone()
        compounds_position[compound] = (x_compound, y_compound, spectrum_found)
    return compounds_position


def filter_calibration_compounds(compounds: dict):
    '''
    Filter compounds which are not in significant list
    :param compounds: dictionary with all compounds
    :return: filtered dictionary
    '''
    global config
    filtered = {}
    for key in compounds:
        if key in config.calibration_compounds:
            filtered[key] = compounds[key]
    return filtered


def apply_color_map(grayscale_image):
    '''
    Apply color map to grayscale image
    :param grayscale_image:
    :return: colored image
    '''
    cm = matplotlib.colormaps['viridis']
    return cm(grayscale_image)[..., :3] * 255


def fill_image(image, ds, data_pointer, scan_beginning_index, scan_length, second_dimension_length):
    '''
    Fill image with data from cdf file
    :param image: image which will be filled with data
    :param ds: loaded cdf file
    :param data_pointer: pointer to data in cdf file
    :param scan_beginning_index: x coordinate where to start filling
    :param scan_length: length of x coordinate
    :param second_dimension_length: length of y coordinate
    :return: updated data_pointer
    '''
    for x in range(scan_beginning_index, scan_beginning_index + scan_length):
        image[:second_dimension_length, x] = ds['total_intensity'][
                                             data_pointer:data_pointer + second_dimension_length].cpu().numpy()
        data_pointer += second_dimension_length

    return data_pointer


def write_spectrum_to_file(spectrum, f):
    '''
    Write spectrum to file
    :param spectrum: spectrum which will be written to file
    :param f: opened file
    :return: None
    '''
    f.write('\t')
    for i in range(len(spectrum)):
        if i == 0:
            continue
        f.write(f'{i}:{spectrum[i]};')
    f.write('\n')


def create_spectrum_graph(dpi, spectrum, title):
    '''
    Create spectrum graph and store it to opened file
    :param dpi: dpi of the graph
    :param spectrum: spectrum which will be plotted
    :param title: title of the graph
    :return: None
    '''
    global args, config
    num_of_points_under_100 = config.constants['m_z_importance_until_100']
    num_of_points_over_100 = config.constants['m_z_importance_until_200']
    num_of_points_over_200 = config.constants['m_z_importance_over_200']
    start = int(config.constants['m_z_start'])
    end = int(config.constants['m_z_end'])
    spectrum_values = spectrum[start:end].cpu().numpy()
    spectrum_values_under_100 = spectrum_values[:100 - start]
    spectrum_values_over_100 = spectrum_values[100 - start:]
    spectrum_values_over_200 = spectrum_values[200 - start:]
    top_indices_under_100 = np.argsort(spectrum_values_under_100)[-num_of_points_under_100:]
    top_indices_over_100 = np.argsort(spectrum_values_over_100)[-num_of_points_over_100:]
    top_indices_over_200 = np.argsort(spectrum_values_over_200)[-num_of_points_over_200:]
    # Create the plot
    plt.figure(figsize=((13, 7)))
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    indices = torch.arange(len(spectrum_values))
    plt.plot(indices + start, spectrum_values, marker='', linestyle='-', color='r', label='Spectrum',
             linewidth=0.8)
    plt.plot(top_indices_under_100 + start, spectrum_values_under_100[top_indices_under_100], marker='o', linestyle='',
             color='g',
             label='Highest Values',
             markersize=1)
    plt.plot(top_indices_over_100 + 100, spectrum_values_over_100[top_indices_over_100], marker='o', linestyle='',
             color='g',
             label='Highest Values',
             markersize=1)
    plt.plot(top_indices_over_200 + 200, spectrum_values_over_200[top_indices_over_200], marker='o', linestyle='',
             color='g',
             label='Highest Values',
             markersize=1)
    for i, txt in enumerate(top_indices_under_100):
        txt = int(txt + start)
        plt.annotate(txt, (top_indices_under_100[i] + start, spectrum_values_under_100[top_indices_under_100[i]]),
                     color='g',
                     ha='right', va='bottom', fontsize=4)
    for i, txt in enumerate(top_indices_over_100):
        txt = int(txt + 100)
        plt.annotate(txt, (top_indices_over_100[i] + 100, spectrum_values_over_100[top_indices_over_100[i]]), color='g',
                     ha='right', va='bottom', fontsize=4)
    for i, txt in enumerate(top_indices_over_200):
        txt = int(txt + 200)
        plt.annotate(txt, (top_indices_over_200[i] + 200, spectrum_values_over_200[top_indices_over_200[i]]), color='g',
                     ha='right', va='bottom', fontsize=4)
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.grid(True)
    cdf_file = args.input_cdf.split('/')[-1]
    cdf_file = cdf_file.replace('.cdf', '')
    save_dir = os.path.join(os.getcwd(), 'tmp', cdf_file)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, title + '.png')
    plt.savefig(save_path, format='png', dpi=dpi)
    plt.close()


def plot_show_maybe_store(viz: np.ndarray, filename: str = None, directory: str = None,
                          dpi: int = 600,
                          invert_color_channels: bool = False, channel_order: str = 'CHW',
                          max_value: float = 255.,
                          aspect: float = 1.0, marker_size: float = 0.05, compounds_pixels: dict = None,
                          compounds_area: dict = None):
    '''
    Plot image and store it
    :param viz:
    :param filename:
    :param directory:
    :param dpi:
    :param invert_color_channels:
    :param channel_order:
    :param max_value:
    :param aspect:
    :param marker_size:
    :param compounds_pixels:
    :return: None
    '''
    global compounds_numbers
    original_height = viz.shape[1]
    original_width = viz.shape[2]
    if channel_order == 'CHW':
        viz = viz.transpose((1, 2, 0))
    elif channel_order != 'HWC':
        raise RuntimeError(f'Unknown channel order {channel_order}!')
    if invert_color_channels:
        viz = viz[..., ::-1]
    viz = np.log(viz + 1e-10)
    min_grather_zero = viz[viz > 0].min()
    viz[viz < 0] = min_grather_zero
    viz = viz / viz.max()
    viz = apply_color_map(viz)
    viz = np.squeeze(viz, axis=2)
    result = np.zeros((viz.shape[0], 5 * viz.shape[1], 3))
    for i in range(5):
        result[:, i::5, :] = viz
    viz = result
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename.replace('.png', '.txt')), 'w') as f:
        if compounds_pixels is not None:
            for compound in compounds_pixels:
                x_compound = compounds_pixels[compound][0]
                y_compound = compounds_pixels[compound][1]
                x_time, y_time = from_pixels_times(x_compound, y_compound)
                x_compound *= 5
                x_compound += 2
                y_compound = original_height - y_compound - 1
                viz = cv2.circle(viz, (x_compound, y_compound), radius=4, color=(255, 0, 0), thickness=2)
                cv2.putText(viz, str(compounds_numbers[compound]), (x_compound + 15, y_compound - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, config.constants['font_size_chromatogram'],
                            (255, 0, 0), config.constants['font_thickness_chromatogram'])
                if compounds_area is not None:
                    f.write(
                        "{}\t{}\t{}\t{}\t{}".format(compounds_numbers[compound], str(compound), str(x_time),
                                                    str(y_time), str(compounds_area[compound])))
                    if args.spectrum == 'yes':
                        write_spectrum_to_file(compounds_pixels[compound][2], f)
                        create_spectrum_graph(dpi, compounds_pixels[compound][2], compound)
                    else:
                        f.write('\n')
                else:
                    f.write(
                        "{}\t{}\t{}\t{}".format(compounds_numbers[compound], str(compound), str(x_time), str(y_time)))
                    if args.spectrum == 'yes':
                        write_spectrum_to_file(compounds_pixels[compound][2], f)
                        create_spectrum_graph(dpi, compounds_pixels[compound][2], compound)
                    else:
                        f.write('\n')
        else:
            raise ValueError('Compounds pixels are not set')
    if args.debug_calibration == 'yes':
        for triangle in triangles:
            v1 = numbers_compounds[triangle[0]]
            v2 = numbers_compounds[triangle[1]]
            v3 = numbers_compounds[triangle[2]]
            x1 = compounds_pixels[v1][0] * 5 + 2
            y1 = original_height - compounds_pixels[v1][1] - 1
            x2 = compounds_pixels[v2][0] * 5 + 2
            y2 = original_height - compounds_pixels[v2][1] - 1
            x3 = compounds_pixels[v3][0] * 5 + 2
            y3 = original_height - compounds_pixels[v3][1] - 1
            viz = cv2.line(viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
            viz = cv2.line(viz, (x2, y2), (x3, y3), (255, 0, 0), 2)
            viz = cv2.line(viz, (x3, y3), (x1, y1), (255, 0, 0), 2)

    cv2.imwrite(os.path.join(directory, filename), viz[..., ::-1])


def visual_check_against_spectrogram(ds, spectrogram_image, directory=None,
                                     filename=None, compounds_pixels=None, compounds_area=None):
    '''
    Visual check of the results against spectrogram
    :param ds:
    :param spectrogram_image:
    :param directory:
    :param filename:
    :param compounds_pixels:
    :return: None
    '''
    global x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, y_ten_seconds, x_ten_seconds, config
    if (config.constants['m_z_index'] == -1):
        spectrogram_image = spectrogram_image.sum(0)
    else:
        spectrogram_image = spectrogram_image[config.constants['m_z_index'], :, :]
    spectrogram_image = spectrogram_image - spectrogram_image[spectrogram_image > 0].min()
    spectrogram_image[spectrogram_image < 0] = 0
    spectrogram_image = spectrogram_image[::-1]
    if filename == 'original.png':
        filename = args.input_cdf.split('/')[-1].replace('.cdf', '.png')
    plot_show_maybe_store(spectrogram_image[np.newaxis], aspect=0.2,
                          directory=directory, filename=filename, compounds_pixels=compounds_pixels,
                          compounds_area=compounds_area)


def load_triangles():
    '''
    Load triangles from yaml file
    :return: None - update global variable triangles
    '''
    global triangles, config
    for i in config.triangles:
        v1, v2, v3 = i.split(' ')
        triangles.append((int(v1), int(v2), int(v3)))


def make_numbers_compounds():
    '''
    Make dictionary with numbers and compounds
    :return: None - update global variable numbers_compounds
    '''
    global numbers_compounds, compounds_numbers
    for compound in compounds_numbers:
        numbers_compounds[compounds_numbers[compound]] = compound


def extract_features(spectrogram_image, compounds_pixels):
    '''
    Extract features for area calculation
    :param spectrogram_image: spectrogram image (3D tensor)
    :param compounds_pixels: dictionary with pixels for each compound
    :return: data for model training
    '''
    global compounds_numbers
    original_number_new_number = {}
    new_number_original_number = {}
    model = {}
    model['feas'] = []
    model['labels_true'] = []
    i = 0
    for compound in compounds_pixels:
        x_compound = compounds_pixels[compound][0]
        y_compound = compounds_pixels[compound][1]
        spectrum = spectrogram_image[:, y_compound:y_compound + 1, x_compound:x_compound + 1].clone().cpu().numpy()
        spectrum = spectrum.flatten()
        model['feas'].append(spectrum)
        model['labels_true'].append(i)
        ref_spectrum = torch.load(f'avg_spectrum/{compound}.pth').cpu().numpy()
        ref_spectrum = ref_spectrum.flatten()
        model['feas'].append(ref_spectrum)
        model['labels_true'].append(i)
        original_number_new_number[compounds_numbers[compound]] = i
        new_number_original_number[i] = compounds_numbers[compound]
        i += 1
    model['labels_true'] = np.array(model['labels_true'])
    model['feas'] = np.array(model['feas'])
    return model


def make_own_cmap():
    '''
    Make own color map for visualization
    :return: color map
    '''
    base_cmaps_20 = ['tab20', 'tab20b', 'tab20c']
    colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2, 0.8, 20)) for name in base_cmaps_20])
    set3_colors = plt.get_cmap('Set3')(np.linspace(0, 1, 20))
    colors = np.concatenate([colors, set3_colors])
    seed = 0
    np.random.seed(seed)
    np.random.shuffle(colors)
    colors = colors[:, :3]
    colors[0] = [0, 0, 0]
    return colors


def visualize_area(image, compounds_pixels):
    '''
    Visualize pixels that are taken into account for area computation
    :param image: image with marked pixels
    :param compounds_pixels: dictionary with pixels for each compound
    :return: None
    '''
    global args, numbers_compounds, config
    colors_rgb = make_own_cmap()
    colors_bgr = colors_rgb.copy()
    fig, ax = plt.subplots(figsize=(1, 4))
    for i in range(len(numbers_compounds)):
        color = colors_rgb[i]
        ax.fill_between([0, 1], i, i + 1, color=color)
        ax.text(1.1, i + 0.5, numbers_compounds[i + 1], va='center',
                fontsize=2)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(colors_bgr))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    colorbar_image = os.path.join(config.constants.output_directory, 'colorbar.png')
    plt.savefig(colorbar_image, bbox_inches='tight', format='png', dpi=500)
    plt.close()
    for i in range(len(colors_bgr)):
        colors_bgr[i, 0], colors_bgr[i, 2] = colors_bgr[i, 2], colors_bgr[i, 0]
        colors_bgr[i] = colors_bgr[i] * 255
    image_bgr = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] != 0:
                image_bgr[i, j, :] = colors_bgr[int(image[i][j] - 1), :]
            else:
                # white
                image_bgr[i, j, :] = [255, 255, 255]
    result = np.zeros((image.shape[0], 5 * image.shape[1], 3))
    for i in range(5):
        result[:, i::5, :] = image_bgr
    image_bgr = result
    image_bgr = cv2.flip(image_bgr, 0)
    for compound in compounds_pixels:
        x_compound = compounds_pixels[compound][0] * 5
        x_compound += 2
        y_compound = image.shape[0] - compounds_pixels[compound][1] - 1
        image_bgr = cv2.putText(image_bgr, str(compounds_numbers[compound]), (x_compound + 15, y_compound - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, config.constants['font_size_chromatogram'], (0, 0, 255),
                                config.constants['font_thickness_chromatogram'])
    save_path_area = os.path.join(config.constants.output_directory, 'area.png')
    cv2.imwrite(save_path_area, image_bgr)
    image1 = Image.open(colorbar_image)
    image2 = Image.open(save_path_area)
    width1, height1 = image1.size
    width2, height2 = image2.size
    new_height = max(height1, height2)
    new_image = Image.new('RGBA', (width1 + width2, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))
    saving_name_merged = 'area_ood.png'
    merged_image_path = os.path.join(config.constants.output_directory, saving_name_merged)
    new_image.save(merged_image_path)
    os.remove(colorbar_image)
    os.remove(save_path_area)


def compute_area(compounds_pixels, spectrogram_image, mask_spectrum, model=None, visualize=True):
    '''
    Compute area for each compound
    :param model: model for area computation
    :param compounds_pixels: dictionary with pixels for each compound
    :param spectrogram_image: spectrogram image (3D tensor)
    :param visualize: specify whether to visualize the result
    :return: dictionary with areas for each compound
    '''
    global config, compounds_numbers
    compounds_area_threshold = OmegaConf.load('compounds_area_threshold.yaml')
    compounds_area = {}
    image = np.zeros((spectrogram_image.shape[1], spectrogram_image.shape[2]))
    image_sens = np.zeros((spectrogram_image.shape[1], spectrogram_image.shape[2]))
    ADD_X = config.constants.area_box_t1
    ADD_Y = config.constants.area_box_t2
    area = 0
    for compound in compounds_pixels:
        sensitivity = compounds_area_threshold[compound]
        mask = mask_spectrum[compound]
        x_compound = compounds_pixels[compound][0]
        y_compound = compounds_pixels[compound][1]
        cut_spetrogram_image = spectrogram_image[:, y_compound - ADD_Y:y_compound + 1 + ADD_Y,
                               x_compound - ADD_X:x_compound + 1 + ADD_X].clone()
        spectrum_at_click = spectrogram_image[:, y_compound:y_compound + 1, x_compound:x_compound + 1].clone()
        for i in range(len(mask)):
            if mask[i] == 0:
                cut_spetrogram_image[i, :, :] = 0
                spectrum_at_click[i, :, :] = 0
        spectrum_at_click = spectrum_at_click.double()
        norma = torch.norm(spectrum_at_click)
        spectrum_at_click = spectrum_at_click / norma
        cut_spetrogram_image = cut_spetrogram_image.double()
        cut_spetrogram_image = cut_spetrogram_image / norma
        # cut_spetrogram_image = torch.nn.functional.normalize(cut_spetrogram_image, dim=0)
        dot = cut_spetrogram_image * spectrum_at_click
        scores = dot.sum(dim=0)
        part_of_image = image[y_compound - ADD_Y:y_compound + 1 + ADD_Y,
                        x_compound - ADD_X:x_compound + 1 + ADD_X].copy()
        part_of_image += scores.cpu().numpy()
        # because of padding
        part_of_image[part_of_image < sensitivity] = 0
        for i in range(part_of_image.shape[0]):
            for j in range(part_of_image.shape[1]):
                if part_of_image[i, j] != 0:
                    if image_sens[y_compound - ADD_Y + i, x_compound - ADD_X + j] == 0 or \
                            image_sens[y_compound - ADD_Y + i, x_compound - ADD_X + j] > part_of_image[i, j]:
                        image_sens[y_compound - ADD_Y + i, x_compound - ADD_X + j] = part_of_image[i, j]
                        image[y_compound - ADD_Y + i, x_compound - ADD_X + j] = compounds_numbers[compound]
        for i in range(part_of_image.shape[0]):
            for j in range(part_of_image.shape[1]):
                if part_of_image[i][j] != 0:
                    area += spectrogram_image[:, i + y_compound - ADD_Y, j + x_compound - ADD_X].sum()
        compounds_area[compound] = area
        for compound in compounds_area:
            if not isinstance(compounds_area[compound], int):
                compounds_area[compound] = int(compounds_area[compound])
        area = 0
    if visualize:
        print('Visualizing area')
        visualize_area(image, compounds_pixels)
    return compounds_area


def load_reference_spectrum_indexes(size):
    '''
    Load important indexes for each compound
    :param size: size of the spectrum
    :return: dictionary with important indexes for each compound
    '''
    compounds_indexes = {}
    with open('compounds_indexes.txt', 'r') as f:
        for line in f:
            compound = line.replace('\n', '')
            indexes = f.readline()
            spectrum = np.zeros(size)
            indexes = indexes.split(' ')
            for index in indexes:
                if index == '\n':
                    break
                index, value = index.split(':')
                spectrum[int(index)] = float(value)
            spectrum[spectrum > 0] = 1
            compounds_indexes[compound] = spectrum
    return compounds_indexes


def main():
    global config, dev, system_number, t1_shift, x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, y_ten_seconds, x_ten_seconds, args, compounds_numbers, triangles
    dev = 'cpu'
    if torch.cuda.is_available():
        dev = 'cuda'
    print(dev)
    parser = argparse.ArgumentParser(
        description='Tool for GCxGC-MS data processing. Made by Jan Hlavsa under supervision of Ing. Bc. Radim Špetlík and prof. Jiří Matas.')
    parser.add_argument('--config', type=str, required=False, default='config.yaml',
                        help='Path to the configuration file (YAML format).')
    parser.add_argument('--input_cdf', type=str, required=False, default='input.cdf',
                        help='Path to the input CDF file.')
    parser.add_argument('--clear', type=str, required=False, choices=['yes', 'no'], default='no',
                        help='Specify whether to clear intermediate data. Choices: yes, no.')
    parser.add_argument('--shift_method', type=str, required=False, choices=['avg', 'triangles'], default='triangles',
                        help='Specify the shift method. Choices: avg, triangles.')
    parser.add_argument('--debug_calibration', type=str, required=False, default='no', choices=['yes', 'no'],
                        help='Enable debug mode for calibration. Plots triangles. Choices: yes, no.')
    parser.add_argument('--area', type=str, required=False, default='no', choices=['yes', 'no'],
                        help='Specify whether to process the area [Work in progress]. Choices: yes, no.')
    parser.add_argument('--spectrum', type=str, required=False, default='no', choices=['yes', 'no'],
                        help='Specify whether to save and plot spectrum. Choices: yes, no.')

    args = parser.parse_args()
    if (args.clear == 'yes'):
        if os.path.exists('tmp'):
            os.system('rm -r tmp')
        if os.path.exists('area_ood.png'):
            os.remove('area_ood.png')
        exit(0)
    cdf_path = args.input_cdf
    config = OmegaConf.load(args.config)
    compounds_numbers = OmegaConf.load('compounds_numbers.yaml')
    make_numbers_compounds()
    load_triangles()
    t1_shift = config.constants.t1_shift
    system_number = config.constants.system_number
    spectrogram_image, ds = load_cdf(cdf_path)
    compounds = load_ref_compounds()
    compounds_pixels = from_times_pixels(compounds)
    avg_shift, calibration_compounds_pixels, shifts_for_compounds = find_shift(compounds_pixels, spectrogram_image)
    if (len(calibration_compounds_pixels.keys()) < len(config.calibration_compounds)):
        print('Not all calibration compounds were found. Please change the calibration compounds as well as triangles.')
        print('Calibration compounds found: ', calibration_compounds_pixels.keys())
        exit(1)
    compounds_positions = find_positions(compounds_pixels, compounds, calibration_compounds_pixels, avg_shift,
                                         spectrogram_image, shifts_for_compounds)
    compounds_area = None
    if args.area == 'yes':
        mask_spectrum = load_reference_spectrum_indexes(spectrogram_image.shape[0])
        # model_outputs = extract_features(spectrogram_image, compounds_positions)
        # dict = {'logits': model_outputs['feas']}
        # msp = ood_msp.MSPOODDetector()
        # msp.setup(args, dict)
        # scores = msp.infer(dict)
        # model = ood_mahalanobis.MahalanobisOODDetector()
        # model.setup(args, model_outputs)
        # scores = model.infer(model_outputs)
        compounds_area = compute_area(compounds_positions, spectrogram_image, mask_spectrum,
                                      visualize=config.constants.visualize_area)
    if args.debug_calibration == 'yes':
        # filter just calibration compounds
        compounds_positions = filter_calibration_compounds(compounds_positions)
    print('Visual check against spectrogram')
    visual_check_against_spectrogram(ds, spectrogram_image.cpu().numpy(),
                                     directory=config.constants.output_directory,
                                     filename=config.constants.output_file_name + '.png',
                                     compounds_pixels=compounds_positions, compounds_area=compounds_area)


if __name__ == '__main__':
    main()
