#!/usr/bin/env bash
cd ~/Downloads
wget --no-check-certificate https://dl.djicdn.com/downloads/manifold/manifold_image_v1.0.tar.gz
mkdir ~/manifold
cd ~/manifold
tar -xvpzf ~/Downloads/manifold_image_v1.0.tar.gz
cd ~/manifold/Linux_for_Tegra
./flash jetson-tk1 mmcblk0p1
