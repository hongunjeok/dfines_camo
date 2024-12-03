import os
from PIL import Image

def compare_results(dir1, dir2, output_dir, mode='horizontal'):
    """
    将两个目录中的同名图片拼接，用于对比算法结果。
    
    :param dir1: 第一个算法结果的目录
    :param dir2: 第二个算法结果的目录
    :param output_dir: 输出拼接图像的目录
    :param mode: 拼接模式，'horizontal' 为水平拼接，'vertical' 为垂直拼接
    """
    os.makedirs(output_dir, exist_ok=True)

    files1 = set(f for f in os.listdir(dir1) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    files2 = set(f for f in os.listdir(dir2) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    
    common_files = files1.intersection(files2)
    if not common_files:
        print("No common files found in the given directories.")
        return

    print(f"Found {len(common_files)} common files. Processing...")
    
    for idx, file_name in enumerate(common_files):
        path1 = os.path.join(dir1, file_name)
        path2 = os.path.join(dir2, file_name)
        
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        
        # 调整两张图片的大小一致
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        # 拼接图片
        if mode == 'horizontal':
            new_img = Image.new("RGB", (img1.width + img2.width, img1.height))
            new_img.paste(img1, (0, 0))
            new_img.paste(img2, (img1.width, 0))
        elif mode == 'vertical':
            new_img = Image.new("RGB", (img1.width, img1.height + img2.height))
            new_img.paste(img1, (0, 0))
            new_img.paste(img2, (0, img1.height))
        else:
            raise ValueError("Mode must be 'horizontal' or 'vertical'.")

        # 保存拼接结果
        save_path = os.path.join(output_dir, f"compare_{file_name}")
        new_img.save(save_path)
        
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(common_files)} files...")

    print(f"All comparison images saved in {output_dir}.")

# 示例运行
if __name__ == "__main__":
    dir1 = './vis_outputs/dfine_x_coco'
    dir2 = './vis_outputs/deim_dfine_hgnetv2_x_coco_50e'
    output_dir = "./vis_outputs/comparisons_dfine_x_deim"  # 拼接结果保存的目录

    dir1 = './vis_outputs/rtdetrv2_r101vd_6x/'
    dir2 = './vis_outputs/deim_rtdetrv2_r101vd_coco_60e'
    output_dir = './vis_outputs/comparisons_rtdetrv2_r101_deim'
    compare_results(dir1, dir2, output_dir, mode='horizontal')  # 可选 'horizontal' 或 'vertical'
