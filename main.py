import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
from utils import AverageMeter
from metrics import dice_coef, iou_score, get_F1, get_accuracy, get_recall, get_precision


from scipy import stats

if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    #  (Ours)
    parse.add_argument('--testpath', type=str,
                       default="")

  # (Ground Truth)
    parse.add_argument('--path', type=str, default="")


    parse.add_argument('--baseline_path', type=str, default="",
                       help="Path to the baseline predictions (e.g., nnU-Net) for P-value comparison")

    args = parse.parse_args()

    for _dataname in ['test']:

    # for _dataname in ['STU']:
        gtpath = os.path.join(args.path, _dataname, "masks")
        prepath = os.path.join(args.testpath, _dataname)


        gtpath = gtpath.replace('\\', '/')
        prepath = prepath.replace('\\', '/')


        basepath = None
        if args.baseline_path is not None:
            basepath = os.path.join(args.baseline_path, _dataname).replace('\\', '/')

        files = os.listdir(gtpath)
        lens = len(files)
        print("##" * 20)
        print(_dataname, "length: ", lens)
        print("##" * 20)


        iou = AverageMeter()
        dice = AverageMeter()
        f1 = AverageMeter()
        acc = AverageMeter()
        recall = AverageMeter()
        precision = AverageMeter()


        dice_list_ours = []
        dice_list_baseline = []

        for file in files:
            gtfile = os.path.join(gtpath, file).replace('\\', '/')
            prefile = os.path.join(prepath, file).replace('\\', '/')


            gt_img = Image.open(gtfile).convert('L')
            pre_img = Image.open(prefile).convert('L')

            gt = np.asarray(gt_img)
            pre = np.asarray(pre_img)

            h, w = gt.shape[:2]

            gt = gt.reshape(h, w, 1) / 255.0
            pre = pre.reshape(h, w, 1) / 255.0

           
            val_dice = dice_coef(pre, gt)  

            iou.update(iou_score(pre, gt))
            dice.update(val_dice)  
            f1.update(get_F1(pre, gt))
            acc.update(get_accuracy(pre, gt))
            recall.update(get_recall(pre, gt))
            precision.update(get_precision(pre, gt))

            
            dice_list_ours.append(val_dice)

            
            if basepath is not None:
                basefile = os.path.join(basepath, file).replace('\\', '/')
                if os.path.exists(basefile):
                    base_img = Image.open(basefile).convert('L')
                    base = np.asarray(base_img)
                    base = base.reshape(h, w, 1) / 255.0

                    
                    val_dice_base = dice_coef(base, gt)
                    dice_list_baseline.append(val_dice_base)
                else:
                    print(f"Warning: Baseline file not found: {file}")
                    
                    dice_list_baseline.append(0.0)

                   
        print("acc:", acc.avg)
        print("Iou", iou.avg)
        print("Dice", dice.avg)
        print("f1", f1.avg)
        print("recall", recall.avg)
        print("precision", precision.avg)

      
        print("-" * 30)
        if basepath is not None and len(dice_list_baseline) == len(dice_list_ours):
            
            try:
                stat, p_value = stats.wilcoxon(dice_list_ours, dice_list_baseline)

                print(f"Statistical Test (Ours vs Baseline):")
                print(f"Baseline Path: {basepath}")
                print(f"Wilcoxon P-value: {p_value:.6f}")  # 保留6位小数

                if p_value < 0.05:
                    print("Result: Significant Difference (p < 0.05) ✅")
                else:
                    print("Result: No Significant Difference (p >= 0.05) ❌")
            except ValueError as e:
                print("Error calculating P-value (possibly identical results):", e)
        else:
            print("P-value calculation skipped: No baseline path provided or length mismatch.")
            print(f"Ours count: {len(dice_list_ours)}")
            print(f"Baseline count: {len(dice_list_baseline) if basepath else 0}")
        print("-" * 30)

        
