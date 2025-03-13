import os
import shutil
import subprocess
import torch
import numpy as np
import argparse

def copy_first_n_images(source_folder, destination_folder, n=10):
    """
    copy the first n images
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # sort all png files
    png_files = sorted([f for f in os.listdir(source_folder) if f.endswith('.png')])
    # only take the first n files
    for filename in png_files[:n]:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy2(source_path, destination_path)



def copy_folder_contents(source_folder, destination_folder):
    """
    copy the contents of the source folder to the destination folder
    """
    # ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # copy files
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)
        
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)

def run_script(script_path, args=None):
    """
    run the specified Python script, optionally adding command line arguments
    """
    # try:
    command = ['python', script_path]
    if args:
        command.extend(args)
    subprocess.run(command, check=True)


def clear_folder(folder_path):
    """
    clear the contents of the specified folder
    """
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

def main():


    # corruptions = ['brightness', 'contrast', 'defocus_blur', 'fog', 'glass_blur', 'jpeg_compression', 'motion_blur', 'pixelate', 'zoom_blur']
    # severitys = ['5','4','3','2','1']

    corruptions = ['defocus_blur']
    severitys = ['3']





    errors_list = []

    log_file = 'Corruption error.log'
    with open(log_file, 'w') as f:
        f.write("Corruption Error Log\n")
        f.write("===================\n\n")

    # data path list
    source_data_list = [
    os.path.join(args.datapath, 'corrupted')
    ]

    # each dataset's virtual images
    testB_list = [
        os.path.join(args.datapath, 'virtual')
    ]


    # each dataset's real images
    train_image_real_images = [
        os.path.join(args.datapath, 'real')
    ]


    pose_gt_file_list = [
        os.path.join(args.datapath, 'pose.txt')
    ]




    for source_data in source_data_list:

        ############### use the first 10 images to initialize the cyclegan ###################################
        # clear the destination folder
        destination_folder = os.path.join(args.datapath, 'trainA')
        target_testB_folder = os.path.join(args.datapath, 'trainB')
        
        print("clear the destination folder...")
        clear_folder(destination_folder)
        clear_folder(target_testB_folder)
        
        # copy the first 10 images
        testB_folder = testB_list[source_data_list.index(source_data)]
        trainA_folder = train_image_real_images[source_data_list.index(source_data)]
        pose_gt_folder = pose_gt_file_list[source_data_list.index(source_data)]
        print("start to copy the first 10 images...")
        copy_first_n_images(trainA_folder, destination_folder, n=10)
        copy_first_n_images(testB_folder, target_testB_folder, n=10)

        # 2. copy the cyclegan model
        source_ckpt_dir = 'pretrained_ckpt'
        target_ckpt_dir = './checkpoints/cyclegan_ckpt'

        os.makedirs(target_ckpt_dir, exist_ok=True)
        for filename in os.listdir(source_ckpt_dir):
            if filename.endswith('.pth'):
                src_path = os.path.join(source_ckpt_dir, filename)
                dst_path = os.path.join(target_ckpt_dir, filename)
                shutil.copy2(src_path, dst_path)



        result_cyclegan_pretrained = subprocess.run(
            [
                'python', 'train_origin.py', '--dataroot', args.datapath,
                 '--name', 'cyclegan_ckpt', '--model', 'cycle_gan',
                '--gpu_ids', '6', '--save_epoch_freq', '9', '--continue_train', '--n_epochs', '0', '--n_epochs_decay', '10',
                '--batch_size', '1', '--dataset_mode', 'unaligned'
            ],
            check=False
        )



        ###################################################################################


        # define the parameters of the baseline script
        baseline_args = [
            '--model', 'cycle_gan',
            '--dataroot', args.datapath,
            '--continue_train',
            '--query_path', os.path.join(args.datapath, 'trainA'),
            '--scene_path', os.path.join(args.datapath, 'trainA'),
            '--pose_gt_path', pose_gt_folder,
            '--name', 'cyclegan_ckpt',
        ]

        for corruption in corruptions:
            for severity in severitys:
                source_folder = os.path.join(source_data,
                corruption, f'{severity}')
                    
                # clear the destination folder
                destination_folder = os.path.join(args.datapath, 'trainA')
                target_testB_folder = os.path.join(args.datapath, 'trainB')
                
                print("clear the destination folder...")
                clear_folder(destination_folder)
                clear_folder(target_testB_folder)
                
                # copy the new files
                testB_folder = testB_list[source_data_list.index(source_data)]
                print("start to copy the folder contents...")
                copy_folder_contents(source_folder, destination_folder)
                copy_folder_contents(testB_folder, target_testB_folder)
                print("the folder contents are copied")

    

                # 3. run r2nav.py
                print("\nbegin running r2nav.py...")
                run_script("r2nav.py", baseline_args)

                # 4. read the error_list_np.npy, and calculate the mean value
                error_list_np = np.load('error_list_np.npy')
                errors_list.append(error_list_np)
                
                mean_error = np.mean(error_list_np)
                median_error = np.median(error_list_np)
                print(f"data path: {source_data}, corruption: {corruption}, severity: {severity}, mean_error: {mean_error}, median_error: {median_error}")
                
                # write the results to the log file
                with open(log_file, 'a') as f:
                    f.write(f"data path: {source_data}\n")
                    f.write(f"Corruption: {corruption}\n")
                    f.write(f"Severity: {severity}\n")
                    f.write(f"Mean Error: {mean_error}\n")
                    f.write(f"Median Error: {median_error}\n")
                    f.write("-" * 30 + "\n")

    # calculate the mean error of all corruptions
    mean_error_list = np.mean(errors_list, axis=0)
    mean_error = np.mean(mean_error_list)

    median_error = np.median(mean_error_list)


    print(f"the mean error of all corruptions: {mean_error}")
    
    # write the overall mean error to the log file
    with open(log_file, 'a') as f:
        f.write("\nOverall Results\n")
        f.write("==============\n")
        f.write(f"Average Error Across All Corruptions: {mean_error}\n")
        f.write(f"Median Error Across All Corruptions: {median_error}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract global features')
    parser.add_argument('--datapath', type=str, default = '/mnt/data/jywu/code/retrieval_transfer_improve/dataset_publish', help='input directory')
    args = parser.parse_args()
    

    main()