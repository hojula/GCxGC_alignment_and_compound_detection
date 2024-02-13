import argparse
from omegaconf import OmegaConf
import netCDF4 as nc
import torch
from tqdm import trange
import os
import cv2
import numpy as np
import matplotlib

config = None

system_number = 0
y_six_seconds = 0
x_six_seconds = 0
y_eight_seconds = 0
x_eight_seconds = 0
y_ten_seconds = 0
x_ten_seconds = 0
t1_shift = 0


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
            mass_data = torch.zeros((mass_max_range_len,), dtype=torch.double).to(dev)
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
    global config
    ds = nc.Dataset(cdf_path)
    if (config.constants.system_number == 1):
        set_system_1(len(ds['total_intensity']))
    elif (config.constants.system_number == 2):
        set_system_2(len(ds['total_intensity']))
    elif (config.constants.system_number == 3):
        set_system_3(len(ds['total_intensity']))
    else:
        raise ValueError('Unknown system number')
    mass_max_range_len = int(ds['mass_range_max'][:].max())
    mass_range_min = 0
    spectrogram_image = torch.zeros((mass_max_range_len, y_ten_seconds,
                                     x_six_seconds + x_eight_seconds + x_ten_seconds),
                                    dtype=torch.double).to(dev)
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
    return spectrogram_image, ds


def load_ref_compounds():
    global system_number
    if (system_number == 1):
        return OmegaConf.load('compounds_system1.yaml')
    elif (system_number == 2):
        return OmegaConf.load('compounds_system2.yaml')
    elif (system_number == 3):
        return OmegaConf.load('compounds_system3.yaml')
    else:
        raise ValueError('Unknown system number')


def calculate_pixels_for_compounds(compounds: dict):
    global x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, y_ten_seconds, x_ten_seconds, system_number, t1_shift
    compounds_pixels = {}
    if system_number == 1:
        # MINIMAL TIME WHEN THE MACHINE IS SET FOR 10 SECONDS IN FIRST COLUMN
        min_ten_seconds = x_six_seconds * 6 + x_eight_seconds * 6
        for compound in compounds:
            # T1 READ FROM FILE
            t1 = compounds[compound][0] + t1_shift
            # T2 READ FROM FILE
            t2 = compounds[compound][1]

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
            t1 = compounds[compound][0] + t1_shift
            # T2 READ FROM FILE
            t2 = compounds[compound][1]

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
    BOX_SIZE_ADD_X = config.constans.box_t1
    BOX_SIZE_ADD_Y = config.constans.box_t2
    avg_shift_x = 0
    avg_shift_y = 0
    counter = 0
    ref_compounds_pixels = {}
    shifts_for_compounds = {}
    for compound in config.constants.significant_compounds:
        spectrum = torch.load(f'avg_spectrum/{compound}.pth')
        x_compound = compounds[compound][0]
        y_compound = compounds[compound][1]
        cut_spectrogram_image = spectrogram_image[:,
                                y_compound - BOX_SIZE_ADD_Y:y_compound + 1 + BOX_SIZE_ADD_Y,
                                x_compound - BOX_SIZE_ADD_X:x_compound + 1 + BOX_SIZE_ADD_X].clone()
        # normalize
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
        y_pos = index_of_max_dot_product // dot_product.shape[1]
        x_pos = index_of_max_dot_product % dot_product.shape[1]
        new_x_compound = x_compound - BOX_SIZE_ADD_X + x_pos
        new_y_compound = y_compound - BOX_SIZE_ADD_Y + y_pos
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
    print(avg_tup)
    del cut_spectrogram_image
    return avg_tup, ref_compounds_pixels, shifts_for_compounds


def find_positions(compounds: dict, ref_compounds_pixels: dict, avg_shift: tuple):
    '''
    Find positions for each compound
    :param compounds: dictionary with compounds and their expected positions pixels
    :param ref_compounds_pixels: dictionary with reference pixels for each significant compound
    :param avg_shift: average shift for all significant compounds
    :return: dictionary with positions for each compound
    '''
    result_dict = {}
    for compound in compounds:
        if (ref_compounds_pixels is not None) and (compound in ref_compounds_pixels):
            x_compound = ref_compounds_pixels[compound][0].item()
            y_compound = ref_compounds_pixels[compound][1].item()
        elif ref_compounds_pixels is None:
            raise ValueError('Reference compounds pixels are not set')
        else:
            x_compound = compounds[compound][0]
            y_compound = compounds[compound][1]
            found = False
            '''
            for triangle in triangles:
                cord = barycentric_coordinates((x_compound, y_compound), triangle, ref_compounds_pixels)
                if cord[0] != -1:
                    found = True
                    shift_x = int(
                        shifts_for_compounds[triangle[0]][0] * cord[0] + shifts_for_compounds[triangle[1]][0] *
                        cord[1] + shifts_for_compounds[triangle[2]][0] * cord[2])
                    shift_y = int(
                        shifts_for_compounds[triangle[0]][1] * cord[0] + shifts_for_compounds[triangle[1]][1] *
                        cord[1] + shifts_for_compounds[triangle[2]][1] * cord[2])
                    x_compound += shift_x
                    y_compound += shift_y
            '''
            if not found:
                x_compound += avg_shift[0]
                y_compound += avg_shift[1]
        result_dict[compound] = (x_compound, y_compound)
    return result_dict


def filter_calibration_compounds(compounds: dict):
    '''
    Filter compounds which are not in significant list
    :param compounds: dictionary with all compounds
    :return: filtered dictionary
    '''
    global config
    filtered = {}
    for key in compounds:
        if key in config.constants.calibration_compounds:
            filtered[key] = compounds[key]
    return filtered


def apply_color_map(grayscale_image):
    cm = matplotlib.colormaps['viridis']
    return cm(grayscale_image)[..., :3] * 255


def fill_image(image, ds, data_pointer, scan_beginning_index, scan_length, second_dimension_length):
    for x in range(scan_beginning_index, scan_beginning_index + scan_length):
        image[:second_dimension_length, x] = ds['total_intensity'][
                                             data_pointer:data_pointer + second_dimension_length].cpu().numpy()
        data_pointer += second_dimension_length

    return data_pointer


def plot_show_maybe_store(viz: np.ndarray, filename: str = None, directory: str = None,
                          dpi: int = 600,
                          invert_color_channels: bool = False, channel_order: str = 'CHW',
                          max_value: float = 255.,
                          aspect: float = 1.0, marker_size: float = 0.05, compounds_pixels: dict = None):
    original_height = viz.shape[1]
    original_width = viz.shape[2]
    print("original_height", original_height)
    print("original_width", original_width)
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
        f.write("{:<8} {:<8} {:<8}\n".format("ID", "t1[px]", "t2[px]"))
        if compounds_pixels is not None:
            for compound in compounds_pixels:
                x_compound = compounds_pixels[compound][0]
                y_compound = compounds_pixels[compound][1]
                x_compound *= 5
                x_compound += 2
                y_compound = original_height - y_compound - 1
                viz = cv2.circle(viz, (x_compound, y_compound), radius=4, color=(255, 0, 0), thickness=3)
                cv2.putText(viz, str(compound), (x_compound + 10, y_compound - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0),
                            2)
                f.write("{:<8} {:<8} {:<8}\n".format(str(compound), str(x_compound), str(y_compound)))
        else:
            raise ValueError('Compounds pixels are not set')
    cv2.imwrite(os.path.join(directory, filename), viz[..., ::-1])


def visual_check_against_total_intensity(ds, spectrogram_image, x_eight_seconds, x_six_seconds,
                                         x_ten_seconds, y_eight_seconds,
                                         y_six_seconds, y_ten_seconds, directory=None,
                                         filename=None, compounds_pixels=None):
    image = np.zeros((y_ten_seconds, x_six_seconds + x_eight_seconds + x_ten_seconds),
                     dtype=np.float32)
    min_ = ds['total_intensity'][:].min()
    min_ = min_.item()
    image += min_

    data_pointer = fill_image(image, ds, 0, 0, x_six_seconds, y_six_seconds)
    data_pointer = fill_image(image, ds, data_pointer, x_six_seconds, x_eight_seconds,
                              y_eight_seconds)
    _ = fill_image(image, ds, data_pointer, x_six_seconds + x_eight_seconds, x_ten_seconds,
                   y_ten_seconds)

    image = image[::-1]

    spectrogram_image = spectrogram_image.sum(0)
    spectrogram_image = spectrogram_image - spectrogram_image[spectrogram_image > 0].min()
    spectrogram_image[spectrogram_image < 0] = 0
    spectrogram_image = spectrogram_image[::-1]

    plot_show_maybe_store(spectrogram_image[np.newaxis], aspect=0.2,
                          directory=directory, filename=filename, compounds_pixels=compounds_pixels)


if __name__ == '__main__':
    dev = 'cpu'
    if torch.cuda.is_available():
        dev = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--input_cdf', type=str, required=True)
    args = parser.parse_args()
    cdf_path = args.input_cdf
    config = OmegaConf.load(args.config)
    print(config)
    t1_shift = config.constants.t1_shift
    system_number = config.constants.system_number
    spectrogram_image, ds = load_cdf(cdf_path)
    # current directory
    dir = os.getcwd()
    compounds = load_ref_compounds()
    compounds_pixels = calculate_pixels_for_compounds(compounds)
    calibration_compounds = filter_calibration_compounds(compounds_pixels)
    avg_shift, calibration_compounds_pixels, shifts_for_compounds = find_shift(compounds_pixels, spectrogram_image)
    compounds_positions = find_positions(compounds_pixels, calibration_compounds_pixels, avg_shift)
    visual_check_against_total_intensity(ds, spectrogram_image, x_eight_seconds, x_six_seconds,
                                         x_ten_seconds, y_eight_seconds,
                                         y_six_seconds, y_ten_seconds, directory=dir,
                                         filename='total_intensity.png', compounds_pixels=compounds_positions)
