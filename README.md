# gleason_web_be
code for graduate degree

# 开发环境
OS: Linux MS-MCRLKPWBBALH 5.10.102.1-microsoft-standard-WSL2 #1 SMP Wed Mar 2 00:30:59 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux

cuda: 11.8

# conda环境和torch
```
conda create -n gleason_web_be python=3.10 -y
conda activate gleason_web_be
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# mambaVision 安装
需要causal_conv1d的可以从下面的网址去下载，这里用的是1.4.0。
文件名的cu118为cuda11.8，cp310即为python 3.10，这两个一定要对上。
下载 [causal_conv1d-1.4.0](https://github.com/Dao-AILab/causal-conv1d/releases) 后放在 /package 目录下，用下面命令安装。
```
pip install package/causal_conv1d-1.4.0+cu118torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install einops huggingface_hub transformers timm
```

mamba_ssm和mambaVsion安装是会报错，这里就不安装了，直接使用github仓库中最新的代码。
执行下面命令，如果不报错则代表mamba安装成功
```
cd demo
python mamba_test.py
python mambaVision_test.py
cd ..
```

# 运行django
```
pip install django django-cors-headers djangorestframework tensorflow matplotlib medmnist
```