import os
import os.path as osp
import argparse
from loguru import logger

def generate_pair_list(mode, output_path):
    """Generate pair list for the dataset.
    
    Args:
        data_root: Dataset root directory
        mode: 'train', 'val', or 'test'
        output_path: Where to save the list file
    """

    pair_dir = osp.join('/Dataset/OSdataset', mode, 'warped', 'image_pair')
    flow_dir = osp.join('/Dataset/OSdataset', mode, 'warped', 'truth_flow')
    
    logger.info(f"Scanning directory: {pair_dir}")
    
    if not osp.exists(pair_dir):
        raise FileNotFoundError(f"Directory not found: {pair_dir}")
    
    opt_files = [f for f in os.listdir(pair_dir) if f.endswith('_opt.tif')]
    
    valid_pairs = []
    for opt_file in sorted(opt_files):
        pair_id = opt_file[:-8]  
        sar_file = f"{pair_id}_sar_warped.tif"
        flow_file = f"{pair_id}.flo"
        
        if (osp.exists(osp.join(pair_dir, sar_file)) and 
            osp.exists(osp.join(flow_dir, flow_file))):
            valid_pairs.append(pair_id)
    
    with open(output_path, 'w') as f:
        for pair_id in valid_pairs:
            f.write(f"{pair_id}\n")
    
    logger.info(f"Generated {len(valid_pairs)} pairs for {mode} mode")
    return valid_pairs

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset utilities')
    parser.add_argument('--task', choices=['generate_lists'], required=True,
                       help='Task to perform')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.task == 'generate_lists':
        for mode in ['train', 'val', 'test']:
            output_path = osp.join(args.data_root, f'{mode}_list.txt')
            try:
                generate_pair_list(mode, output_path)
            except FileNotFoundError as e:
                logger.error(f"Error processing {mode} mode: {str(e)}")
                continue

if __name__ == '__main__':
    main()