#!/bin/bash

echo "开始下载Anaconda安装脚本"
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O ~/anaconda_installer.sh

echo "安装Anaconda到用户目录下的anaconda3文件夹"
bash ~/anaconda_installer.sh -b -p $HOME/anaconda3

echo "删除安装脚本"
rm ~/anaconda_installer.sh

echo "将Anaconda添加到环境变量"
export PATH="$HOME/anaconda3/bin:$PATH"

echo "初始化conda"
conda init bash

echo "conda安装成功！"