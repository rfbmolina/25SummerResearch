import pandas as pd
import numpy as np

def load_data(csv_path: str, chunksize: int = 50_000):
    
    #Load the entire file path:
      #keep every row
      #keep every column (5 meta + all feat_*)
      #cast feat_* to float32 and Class to int8
    #Returns
    #X: DataFrame  (only feat_* columns, float32)
    #y: Series     (Class, int8)
    #groups: Series (Info_group, original dtype)
    

    
    # Discover column names from the header
    all_cols = pd.read_csv(csv_path, sep=";", nrows=0).columns.tolist()

    meta_cols = [
        # "Info_PepID", (Didn't use these, but in case they are needed)
        # "Info_pos",
        # "Info_split",
        "Info_group",
        "Class",
    ]

    #All feature columns start with "feat_"
    feat_cols = [c for c in all_cols if c.startswith("feat_")]
    usecols   = meta_cols + feat_cols
    print(f"→ Loading columns: {len(usecols)} "
          f"(5 meta + {len(feat_cols)} features)")

    #dtype map – only numeric feat_* columns to float32
    dtype_map = {c: "float32" for c in feat_cols}
    dtype_map["Class"] = "int8"      # label column to int8

    
    # Stream the CSV in chunks and concatenate
    reader = pd.read_csv(csv_path,
                         sep=";", quotechar='"', engine="python",
                         usecols=usecols, dtype=dtype_map,
                         chunksize=chunksize, on_bad_lines="skip")

    df = pd.concat(reader, ignore_index=True)
    print(f"→ Loaded DataFrame shape: {df.shape}")

    # Split into X, y, groups
    y       = df.pop("Class").astype(np.int8)
    groups  = df.pop("Info_group")
    # df.drop(columns=["Info_PepID", "Info_pos", "Info_split"], inplace=True)

    X = df  # only feat_* columns remain
    print(f"X: {X.shape}, y: {y.shape}, groups: {groups.shape}")
    return X, y, groups
