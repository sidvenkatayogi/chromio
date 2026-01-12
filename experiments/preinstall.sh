#!/bin/bash

DEST_DIR="data"
VF_FILE_ID="1UbucncHTHGCrMGtML5q5gi_Tae5zFaq4"
VF_FILE_NAME="hexcolor_vf.zip"

wget --no-check-certificate "https://drive.google.com/uc?export=download&id=$VF_FILE_ID" -O "$VF_FILE_NAME"
unzip -q "$VF_FILE_NAME"
mkdir -p "$DEST_DIR"

rm -rf "$DEST_DIR/hexcolor_vf"
mv "hexcolor_vf" "$DEST_DIR/"
rm "$VF_FILE_NAME"

echo "Preinstall script completed."