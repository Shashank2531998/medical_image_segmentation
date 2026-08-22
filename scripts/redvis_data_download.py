import redivis
import os

# 1. Define the Redivis references
user = redivis.user("aimi")
dataset = user.dataset("skm_tea:5r8z:v1_0")
table = dataset.table("image_files:jnty")

# 2. Set up the download directory
download_dir = "/home/woody/iwi5/iwi5326h/projects/VoxTell/data/skm_tea/h5_img_files"
os.makedirs(download_dir, exist_ok=True)

print("Fetching the table data...")
# 3. Load the full table into a DataFrame (no max_results limit)
df = table.to_pandas_dataframe()

total_files = len(df)
print(f"Found {total_files} files. Starting download...")

# 4. Iterate through every row and download the files
for index, row in df.iterrows():
    file_name = row['file_name']
    
    print(f"Downloading ({index + 1}/{total_files}): {file_name}")
    
    # Reference the file by its name and download it
    table.file(file_name).download(download_dir)

print(f"\nSuccess! All files downloaded to: {os.path.abspath(download_dir)}")
