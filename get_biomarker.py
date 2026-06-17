from pathlib import Path
import cv2
import numpy as np
import os
import math
from tqdm import tqdm
import traceback
from skimage.morphology import skeletonize, medial_axis  # 骨架化函数
import matplotlib.pyplot as plt


def get_od_max_circle(od_mask):
    """
    Args:
        od_mask (np.ndarray): 视盘二值掩码图像
        
    Returns:
        tuple: 
            - (cx, cy) (tuple[int, int]): 外接圆中心坐标
            - dd (float): 视盘直径
    """

    contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0), 0.0
    
    max_contour = max(contours, key=cv2.contourArea)

    (cx, cy), radius = cv2.minEnclosingCircle(max_contour)
    dd = 2 * radius
    
    return (int(cx), int(cy)), dd


def generate_annular_masks(av_img, od_center, dd):
    """
    Args:
        av_img (np.ndarray): 动静脉分割图像
        od_center (tuple[int, int]): 视盘中心坐标
        dd (float): 视盘直径
        
    Returns:
        tuple: 
            - a_mask (np.ndarray): A区掩码
            - b_mask (np.ndarray): B区掩码
            - c_mask (np.ndarray): C区掩码
    """
    h, w = av_img.shape[:2]
    cx, cy = od_center
    
    a_mask = np.zeros((h, w), dtype=np.uint8)
    b_mask = np.zeros((h, w), dtype=np.uint8)
    c_mask = np.zeros((h, w), dtype=np.uint8)
    
    od_radius = dd / 2
    a_outer_radius = od_radius + 0.5 * dd
    b_outer_radius = od_radius + 1.0 * dd
    c_outer_radius = od_radius + 2.0 * dd
    
    cv2.circle(a_mask, (cx, cy), int(a_outer_radius), 255, -1)
    cv2.circle(a_mask, (cx, cy), int(od_radius), 0, -1)
    
    cv2.circle(b_mask, (cx, cy), int(b_outer_radius), 255, -1)
    cv2.circle(b_mask, (cx, cy), int(a_outer_radius), 0, -1)
    
    cv2.circle(c_mask, (cx, cy), int(c_outer_radius), 255, -1)
    cv2.circle(c_mask, (cx, cy), int(b_outer_radius), 0, -1)
    
    return a_mask, b_mask, c_mask


def get_top_n_vessels_in_c(vessel_mask, c_mask, top_n=6):
    """
    Args:
        vessel_mask (np.ndarray): 血管二值掩码 (uint8)
        c_mask (np.ndarray): C区掩码 (uint8)
        top_n (int): 选取最粗血管的数量
        
    Returns:
        list[float]: 最粗N段血管的最大直径列表
    """
    
    vessel_in_c = cv2.bitwise_and(vessel_mask, vessel_mask, mask=c_mask)
    
    _, bin_mask = cv2.threshold(vessel_in_c, 127, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8, cv2.CV_32S)
    vessels = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        vessels.append( ( -area, x, y, w, h ) )
    
    vessels_sorted = sorted(vessels)[:top_n]
    diameters = []
    for v in vessels_sorted:
        area_neg, x, y, w, h = v
        area = -area_neg
        mask_roi = np.zeros_like(vessel_mask)
        mask_roi[y:y+h, x:x+w] = 255
        vessel_roi = cv2.bitwise_and(vessel_mask, mask_roi)
   
        skeleton, dist = medial_axis(vessel_roi, return_distance=True)
        vessel_diameters = dist[skeleton] * 2
        max_diameter = vessel_diameters.max()
        d = max_diameter
        diameters.append(d)
    if len(diameters) < top_n:
        diameters += [0.0] * (top_n - len(diameters))
    return diameters


def calculate_crae_crve_revised(vessel_areas, is_artery = True):
    """
    Args:
        vessel_areas (list[float]): C区内最粗6段血管的直径列表
        is_artery (bool): True=计算CRAE,False=计算CRVE
        
    Returns:
        float: CRAE/CRVE计算结果
    """
    
    coeff = 0.88 if is_artery else 0.95

    values = sorted(vessel_areas, reverse=True)

    while len(values) > 1:
        values = sorted(values, reverse=True)
        next_values = []
        i = 0
        j = len(values) - 1

        while i < j:
            w1 = values[i]
            w2 = values[j]
            w_new = coeff * math.sqrt(w1 ** 2 + w2 ** 2)
            next_values.append(w_new)
            i += 1
            j -= 1

        values = next_values

    return values[0]


def calculate_density_in_c(vessel_mask, c_mask):
    """
    Args:
        vessel_mask (np.ndarray): 血管二值掩码
        c_mask (np.ndarray): C区掩码
        
    Returns:
        float: 血管密度值
    """

    _, vessel_bin = cv2.threshold(vessel_mask, 127, 1, cv2.THRESH_BINARY)
    _, c_bin = cv2.threshold(c_mask, 127, 1, cv2.THRESH_BINARY)
    
    vessel_in_c = vessel_bin * c_bin
    vessel_pixels = np.sum(vessel_in_c)
    c_pixels = np.sum(c_bin)
    
    if c_pixels == 0:
        return 0.0
    
    return vessel_pixels / c_pixels


def calculate_fractal_dimension_skeleton(binary_img):
    """
    Args:
        binary_img (np.ndarray): 血管二值掩码 (uint8)
        
    Returns:
        float: 分形维数值
    """
    
    if binary_img.max() == 0:
        return 0.0
    
    _, binary = cv2.threshold(binary_img, 127, 1, cv2.THRESH_BINARY)
    
    skeleton = skeletonize(binary).astype(np.uint8)

    rows, cols = skeleton.shape
    max_box_size = min(rows, cols) // 2
    min_box_size = 1
    
    box_sizes = []
    box_counts = []
    #box_size = min_box_size
    
    for box_size in range(min_box_size, max_box_size + 1):
        
        count = 0

        for i in range(0, rows, box_size):
            for j in range(0, cols, box_size):

                i_end = min(i + box_size, rows)
                j_end = min(j + box_size, cols)
                
                if np.sum(skeleton[i:i_end, j:j_end]) > 0:
                    count += 1
        
        if count > 0:
            box_sizes.append(math.log(1.0 / box_size))
            box_counts.append(math.log(count))
       
    if len(box_sizes) < 2:
        return 0.0
    
    coeffs = np.polyfit(box_sizes, box_counts, 1)

    return coeffs[0]


def extract_av_masks(av_img):
    """
    Args:
        av_img (np.ndarray): 动静脉分割RGB图像
        
    Returns:
        tuple:
            - artery_mask (np.ndarray): 动脉二值掩码
            - vein_mask (np.ndarray): 静脉二值掩码
    """
    
    r_channel = av_img[:, :, 0]
    g_channel = av_img[:, :, 1]
    b_channel = av_img[:, :, 2]
    
    artery_mask = np.logical_and(g_channel, ~b_channel).astype(np.uint8) * 255
    vein_mask = np.logical_and(g_channel, ~r_channel).astype(np.uint8) * 255
    return artery_mask, vein_mask


def process_av_indicators(av_dir, disc_dir, output_dir):
    """
    计算动静脉7项指标并保存结果
    指标列表：
    1. CRAE - 视网膜中央动脉当量
    2. CRVE - 视网膜中央静脉当量
    3. AVR  - 动静脉比 (CRAE/CRVE)
    4. artery_density - C区动脉密度
    5. vein_density - C区静脉密度
    6. artery_fractal_dimension - 动脉分形维数
    7. vein_fractal_dimension - 静脉分形维数
    
    Args:
        av_dir (str): 动静脉分割图像文件夹路径
        disc_dir (str): 视盘轮廓图像文件夹路径
        output_dir (str): 结果文件保存文件夹路径
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    av_suffixes = ('.png', '.PNG')
    av_files = [f for f in os.listdir(av_dir) if f.lower().endswith(av_suffixes)]
    
    for fname in tqdm(av_files, desc="Calculating AV indicators"):
        try:
            av_path = os.path.join(av_dir, fname)
            disc_path = os.path.join(disc_dir, fname)
            txt_path = os.path.join(output_dir, Path(fname).stem + '.txt')
            
            if not os.path.exists(disc_path):
                print(f"Warning: Disc file {fname} not found, skip")
                continue
            
            av_img = cv2.imread(av_path, cv2.IMREAD_COLOR)
            av_img = cv2.cvtColor(av_img, cv2.COLOR_BGR2RGB)
            disc_img = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)
            
            if av_img is None or disc_img is None:
                print(f"Warning: Failed to read {fname}, skip")
                continue
            
            if av_img.shape[:2] != disc_img.shape:
                disc_img = cv2.resize(disc_img, (av_img.shape[1], av_img.shape[0]))
            
            artery_mask, vein_mask = extract_av_masks(av_img)
            
            _, od_bin = cv2.threshold(disc_img, 200, 255, cv2.THRESH_BINARY)
            
            od_center, dd = get_od_max_circle(od_bin)
            if dd == 0:
                print(f"Warning: OD circle not found for {fname}, skip")
                continue
            
            _, _, c_mask = generate_annular_masks(av_img, od_center, dd)
            
            top6_artery_areas = get_top_n_vessels_in_c(artery_mask, c_mask, top_n=6)
            top6_vein_areas = get_top_n_vessels_in_c(vein_mask, c_mask, top_n=6)
            
            crae = calculate_crae_crve_revised(top6_artery_areas, is_artery = True)
            crve = calculate_crae_crve_revised(top6_vein_areas, is_artery = False)
            
            avr = crae / crve if crve > 0 and crae > 0 else float('inf')
            
            artery_density = calculate_density_in_c(artery_mask, c_mask)
            vein_density = calculate_density_in_c(vein_mask, c_mask)
            
            artery_fractal = calculate_fractal_dimension_skeleton(artery_mask)
            vein_fractal = calculate_fractal_dimension_skeleton(vein_mask)
            
            results = {
                "CRAE": crae,
                "CRVE": crve,
                "AVR": avr,
                "artery_density": artery_density,
                "vein_density": vein_density,
                "artery_fractal_dimension": artery_fractal,
                "vein_fractal_dimension": vein_fractal
            }
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                for idx, (key, value) in enumerate(results.items(), 1):
                    if isinstance(value, float):
                        if math.isinf(value):
                            f.write(f"{key} N/A (分母为0)\n")
                        else:
                            f.write(f"{key} {value:.6f}\n")
                    else:
                        f.write(f"{key} {value}\n")
            
            print(f"Successfully saved results to {txt_path}")
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            print(traceback.format_exc())
            continue


if __name__ == "__main__":
    # 动静脉分割图像文件夹
    AV_DIR = r"/home/clara/R2-V2_materials/__predictions/baseline_av/av"
    # 视盘轮廓图像文件夹
    DISC_DIR = r"/home/Data/GAVE2/validation/masks_OD"
    # 结果文件保存文件夹
    OUTPUT_DIR = r"/home/clara/R2-V2_materials/__predictions/baseline_av/biomarkers"
    
    process_av_indicators(AV_DIR, DISC_DIR, OUTPUT_DIR)