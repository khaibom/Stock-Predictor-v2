import os

def save_data(df, filename, dir, context):
    os.makedirs(dir, exist_ok=True)
    path = f"{dir}/{filename}"
    df.to_csv(path, index=False)
    context.log.info(f"Saved data to {path}")