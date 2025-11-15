import os

def save_data(df, filename, dir, context, asset):
    os.makedirs(dir, exist_ok=True)
    path = f"{dir}/{filename}"
    df.to_csv(path, index=False)
    context.log.info(f"Saved {asset} to {path}")