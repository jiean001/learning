#!/bin/sh

# author: jiean001
# 2018-07.06

# 将一个label的image放在一起

current_dir=$1
current_label=$2

echo "current_dir: {$current_dir}, current_label: {$current_label}"

if [ ! -d $current_dir/$current_label ];then
mkdir $current_dir/$current_label
else
echo "{$current_dir/$current_label} ..."
fi

cd $current_dir
mv *_$current_label.png ./$current_label

# echo mv $current_dir/*_$current_label.png $current_dir/$current_label
# mv $current_dir/*_$current_label.png $current_dir/$current_label