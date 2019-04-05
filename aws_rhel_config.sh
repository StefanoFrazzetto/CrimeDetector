#!/user/bin/env bash

# Check user privileges
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

yum install epel-release -y

# Python 3 & pip
yum install python3 -y
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

# Git & Git LFS
yum install git -y
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
yum install git-lfs -y

# Enable LFS and pull
git lfs enable
git lfs pull

echo "All done! Remember to run the project files using python3."
