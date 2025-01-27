import subprocess

def run_python_files(file_list):
  for file in file_list:
    print(f"Running {file}...")
    result = subprocess.run(['python', file], capture_output=True, text=True)

if __name__ == "__main__":
    # 列出你需要运行的Python文件
    files_to_run = [
      'hashing.py','hashing_dpsh_resnet50_nus_wide_32.py', 'hashing_dpsh_resnet34_nus_wide_64.py', 'hashing_dpsh_vgg11_nus_wide_64.py',
      'hashing_csq_resnet50_nus_wide_64.py','hashing_csq_resnet50_nus_wide_32.py', 'hashing_csq_resnet34_nus_wide_64.py', 'hashing_csq_vgg11_nus_wide_64.py',
      'hashing_hashnet_resnet50_nus_wide_64.py','hashing_hashnet_resnet50_nus_wide_32.py', 'hashing_hashnet_resnet34_nus_wide_64.py', 'hashing_hashnet_vgg11_nus_wide_64.py',
      ]
    run_python_files(files_to_run)