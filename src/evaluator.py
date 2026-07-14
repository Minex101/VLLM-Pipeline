import pandas as pd
import re

GT_PATH = "./dataset/dataset.xlsx"
PRED_PATH = "./results.csv"

def parse_dimensions(dim_str):
    return [float(p) for p in str(dim_str).split('(')[0].split(' x ')]

def mean(vals):
    return sum(vals) / len(vals)

def median(vals):
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]

def calculate_absolute_error(GT_result, pred_rows):
    errors = []
    for idx, row in enumerate(pred_rows):
        if row[0] == GT_result[idx][0]:
            try:
                image_name, pred_weight, pred_length, pred_width, pred_height = row
                pred_weight = float(pred_weight)
                pred_length = float(pred_length)
                pred_width = float(pred_width)
                pred_height = float(pred_height)
            except ValueError:
                print(f"Skipping bad row: {row}")
                continue

            _, gt_weight, gt_length, gt_width, gt_height = GT_result[idx]

            # Absolute Errors (for MAE)
            weight_error = abs(pred_weight - gt_weight)
            length_error = abs(pred_length - gt_length)
            width_error = abs(pred_width - gt_width)
            height_error = abs(pred_height - gt_height)

            # percentage errors (for MAPE)
            weight_pct = weight_error / gt_weight * 100
            length_pct = length_error / gt_length * 100
            width_pct = width_error / gt_width * 100
            height_pct = height_error / gt_height * 100

            errors.append([
                image_name,
                weight_error, length_error, width_error, height_error,
                weight_pct, length_pct, width_pct, height_pct
            ])
    return errors

# ---

def main():
    # Read GT excel
    gt_df = pd.read_excel(
        GT_PATH, 
        sheet_name="Objects", 
        header=2, 
        usecols=['Image File', 'Mass (g)', 'Dimensions (cm)']
    )
    
    # Get list of lists format [['images/B00ESJGNBO_51w+0e-eqdL.jpg', 33.9, 13.0, 6.2, 1.19], [..]..]
    rows = gt_df.values.tolist()
    GT_result = []
    for row in rows:
        mass_g, dims_str, image_file = row
        length, width, height = parse_dimensions(dims_str)
        GT_result.append([image_file, float(mass_g), length, width, height])
    
    # Read the predictions excel
    pred_df = pd.read_csv(PRED_PATH)
    pred_rows = pred_df.values.tolist()

    errors = calculate_absolute_error(GT_result, pred_rows)

    weight_errors = [e[1] for e in errors]
    length_errors = [e[2] for e in errors]
    width_errors  = [e[3] for e in errors]
    height_errors = [e[4] for e in errors]

    weight_pct = [e[5] for e in errors]
    length_pct = [e[6] for e in errors]
    width_pct  = [e[7] for e in errors]
    height_pct = [e[8] for e in errors]

    fields = {
        "weight_g": (weight_errors, weight_pct),
        "length_cm": (length_errors, length_pct),
        "width_cm": (width_errors, width_pct),
        "height_cm": (height_errors, height_pct),
    }

    for field, (abs_err, pct_err) in fields.items():
        print(f"\n{field}")
        print(f"  MAE:          {mean(abs_err):.2f}")
        print(f"  Median AE:    {median(abs_err):.2f}")
        print(f"  MAPE:         {mean(pct_err):.2f}%")

if __name__ == "__main__":
    main()