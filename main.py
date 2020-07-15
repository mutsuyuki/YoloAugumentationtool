#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------
# usage: python main.py xxxx 
# ---------------------------------------------------------------
#
#  template.names.txt にクラス名一覧を入れてから実施 ！
#  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#  in xxxx_images/  xxxx_labels_json/
#  out xxxx_labels_txt/ xxxx_my_train_data/
#
# #

import cv2
import numpy as np
import sys
import os
import json


# ヒストグラム均一化
def equalizeHistRGB(src):
    RGB = cv2.split(src)
    Blue = RGB[0]
    Green = RGB[1]
    Red = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])

    img_hist = cv2.merge([RGB[0], RGB[1], RGB[2]])
    return img_hist

# ガウシアンノイズ
def addGaussianNoise(src):
    row, col, ch = src.shape
    mean = 0
    var = 0.05
    sigma = 5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = src + gauss

    return noisy

# salt&pepperノイズ
def addSaltPepperNoise(src):
    row, col, ch = src.shape
    s_vs_p = 0.5
    amount = 0.004
    out = src.copy()
    # Salt mode
    num_salt = np.ceil(amount * src.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in src.shape]
    out[coords[:-1]] = (255, 255, 255)

    # Pepper mode
    num_pepper = np.ceil(amount * src.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in src.shape]
    out[coords[:-1]] = (0, 0, 0)
    return out

# グレースケール変換
def rgbaToGray(src):
    return cv2.cvtColor(src, cv2.COLOR_RGBA2GRAY)



if __name__ == '__main__':

    with open("template_names.txt", encoding="utf-8") as rf:
        print ("----------- template_names.txt -------------")
        print ( rf.read() )
        print ("--------------------------------------------")

    inp=input('Are classes correct ? [y/n]')
    if inp!='y' and inp!='Y':
        exit()


    # プロジェクト名からパスを構築
    dirNameBase = sys.argv[1]
    print("\ntarget project is " + dirNameBase + "\n")

    inImageDir = dirNameBase + "_images/"
    inLabelJsonDir = dirNameBase + "_labels_json/"
    inLabelTxtDir = dirNameBase + "_labels_txt/"

    outDir = "./" + dirNameBase + "_train_data/" 
    outImageDir = outDir + "/JPEGImages/"
    outLabelDir = outDir + "/labels/"

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    if not os.path.exists(outImageDir):
        os.makedirs(outImageDir)
    
    if not os.path.exists(outLabelDir):
        os.makedirs(outLabelDir)


    # JSON --> yolo教師ファイルへ
    print("start converting json to yolo teacher file")
    jsonFiles = os.listdir(inLabelJsonDir)
    for jsonName in jsonFiles:
        file_path = inImageDir + jsonName
        file_type = os.path.splitext(os.path.basename(file_path))[1]
        basename = os.path.splitext(os.path.basename(file_path))[0]

        if(not os.path.exists(inLabelJsonDir + basename + ".json")):
            continue

        with open(inLabelJsonDir + basename + ".json", encoding="utf-8") as f:
            jsonDic = json.load(f)
            teacher_data = []
            for jsonRec in jsonDic:
                bounding = jsonDic[jsonRec]['bounding']
                centerX = bounding['left'] + (bounding['width'] / 2)
                centerY = bounding['top'] + (bounding['height'] / 2)
                width = bounding['width']
                height = bounding['height']
                teacher_data.append( jsonDic[jsonRec]['class'] + " " + str(centerX) + " " + str(centerY) + " " + str(width) + " " + str(height) +"\n")

        with open(inLabelTxtDir + basename + ".txt", "w", encoding="utf-8") as f:
            for line in teacher_data:
                f.write(line)

    print("finish converting json to yolo teacher file")

    # ルックアップテーブルの生成
    min_table = 20
    max_table = 255
    diff_table = max_table - min_table
    gamma1 = 0.60
    gamma2 = 1.1

    LUT_HC = np.arange(256, dtype='uint8')
    LUT_LC = np.arange(256, dtype='uint8')
    LUT_G1 = np.arange(256, dtype='uint8')
    LUT_G2 = np.arange(256, dtype='uint8')

    LUTs = []

    # 平滑化用
    average_square = (2, 2)

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0

    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table

    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # その他LUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255
        LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
        LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

    LUTs.append(LUT_HC)
    LUTs.append(LUT_LC)
    LUTs.append(LUT_G1)
    LUTs.append(LUT_G2)


    files = os.listdir(inImageDir)

    for file_name in files:
        file_path = inImageDir + file_name
        file_type = os.path.splitext(os.path.basename(file_path))[1]
        basename = os.path.splitext(os.path.basename(file_path))[0]

        if(not os.path.exists(inLabelTxtDir + basename + ".txt")):
            continue

        if(file_type != ".jpg" and file_type != ".png" and file_type != ".JPG" and file_type != ".PNG" ):
            continue

        print(file_path)
        # 画像の読み込み
        img_src = cv2.imread(file_path, 1)
        trans_img = []
        trans_img.append(img_src)

        # LUT変換
        for i, LUT in enumerate(LUTs):
            trans_img.append(cv2.LUT(img_src, LUT))

        # 平滑化
        trans_img.append(cv2.blur(img_src, average_square))

        # ヒストグラム均一化
        # trans_img.append(equalizeHistRGB(img_src))

        # ノイズ付加
        trans_img.append(addGaussianNoise(img_src))
        trans_img.append(addSaltPepperNoise(img_src))

        # グレースケール変換
        # trans_img.append(rgbaToGray(img_src))

        # 反転
        flip_horizon_img = []
        for img in trans_img:
            flip_horizon_img.append(cv2.flip(img, 1))

        flip_vertical_img = []
        for img in trans_img:
            flip_vertical_img.append(cv2.flip(img, 0))


        img_src.astype(np.float64)
        print(len(trans_img))
       
        # 通常
        for i, img in enumerate(trans_img):
            cv2.imwrite(outImageDir + basename + "_" + str(i) + ".jpg", img)
            teacher_data = []
            with open(inLabelTxtDir + basename + ".txt",encoding="utf-8") as f:
                for line in f:
                    data = line.split(" ")
                    if (len(data) < 5):
                        break
                    teacher_data.append( data[0] + " " + data[1] + " " + data[2] + " " + data[3] + " " + data[4].replace("\n", "") +"\n")

            with open(outLabelDir + basename + "_" + str(i) + ".txt", "w", encoding="utf-8") as f:
                file_index = i + len(trans_img)
                for line in teacher_data:
                    f.write(line)

        # 横反転
        for i, img in enumerate(flip_horizon_img):
            file_index = i + len(trans_img)
            cv2.imwrite(outImageDir + basename + "_" + str(file_index) + ".jpg", img)
            teacher_data = []
            with open(inLabelTxtDir + basename + ".txt", encoding="utf-8") as f:
                for line in f:
                    data = line.split(" ")
                    if (len(data) < 5):
                        break
                    teacher_data.append( data[0] + " " + str(1 - float(data[1])) + " " + data[2] + " " + data[3] + " " + data[4].replace("\n", "") +"\n")

            with open(outLabelDir + basename + "_" + str(file_index) + ".txt", "w", encoding="utf-8") as f:
                file_index = i + len(trans_img)
                for line in teacher_data:
                    f.write(line)

        # 縦反転
        for i, img in enumerate(flip_vertical_img):
            file_index = i + len(trans_img) * 2
            cv2.imwrite(outImageDir + basename + "_" + str(file_index) + ".jpg", img)
            teacher_data = []
            with open(inLabelTxtDir + basename + ".txt", encoding="utf-8") as f:
                for line in f:
                    data = line.split(" ")
                    if (len(data) < 5):
                        break
                    teacher_data.append( data[0] + " " + data[1] + " " + str(1 - float(data[2])) + " " + data[3] + " " + data[4].replace("\n", "") +"\n")

            with open(outLabelDir + basename + "_" + str(file_index) + ".txt", "w", encoding="utf-8") as f:
                file_index = i + len(trans_img)
                for line in teacher_data:
                    f.write(line)
        


    # ---------- 共通 settings ------------
    # 画像パス一覧ファイルを作成
    outDataListPath = outDir + "data_list.txt"
    createdImagefiles = os.listdir(outImageDir)
    with open(outDir + 'data_list.txt', 'w', newline="\n", encoding="utf-8") as f:
        for file_name in createdImagefiles:
            f.write(outImageDir + file_name + "\n")

    # namesファイルを作成
    outNameFilePath = outDir + "names.txt"
    with open("template_names.txt", encoding="utf-8") as rf:
        with open(outNameFilePath, "w", newline="\n",encoding="utf-8") as wf:
            wf.write(rf.read())
    
    # クラス数取得
    nameLength = sum(1 for line in open("template_names.txt"))

    # ---------- yolo settings ------------
    # cfgファイルを作成 (yolo v2)
    outV2CfgPath = outDir + "v2_cfg.txt"
    with open("template_v2_cfg.txt", encoding="utf-8") as rf:
        with open(outV2CfgPath, "w", newline="\n",encoding="utf-8") as wf:
            cfg = rf.read()
            cfg = cfg.replace("FILTERS_NUM", str((5 + nameLength) * 5 ))
            cfg = cfg.replace("CLASS_NUM", str(nameLength))
            wf.write(cfg)

    # cfgファイルを作成 (yolo v3)
    outV3CfgPath = outDir + "v3_cfg.txt"
    with open("template_v3_cfg.txt", encoding="utf-8") as rf:
        with open(outV3CfgPath, "w", newline="\n",encoding="utf-8") as wf:
            cfg = rf.read()
            cfg = cfg.replace("FILTERS_NUM", str((5 + nameLength) * 3 ))
            cfg = cfg.replace("CLASS_NUM", str(nameLength))
            wf.write(cfg)
 
    # cfgファイルを作成 (yolo v4)
    outV4CfgPath = outDir + "v4_cfg.txt"
    with open("template_v4_cfg.txt", encoding="utf-8") as rf:
        with open(outV4CfgPath, "w", newline="\n",encoding="utf-8") as wf:
            cfg = rf.read()
            cfg = cfg.replace("FILTERS_NUM", str((5 + nameLength) * 3 ))
            cfg = cfg.replace("CLASS_NUM", str(nameLength))
            wf.write(cfg)
 
    # 学習済みモデル保存先フォルダ作成 (yolo)
    modelSaveDir = outDir + "model_saved"
    if not os.path.exists(modelSaveDir):
        os.makedirs(modelSaveDir)

    # dataファイルを作成 (yolo)
    outDataPath = outDir + "data.txt"
    with open("template_data.txt", encoding="utf-8") as rf:
         with open(outDataPath, "w", newline="\n",encoding="utf-8") as wf:
            data = rf.read()
            data = data.replace("CLASS_NUM", str(nameLength))
            data = data.replace("DATA_LIST_PATH", outDataListPath )
            data = data.replace("NAME_LIST_PATH", outNameFilePath)
            data = data.replace("BACKUP_DIR", modelSaveDir)
            wf.write(data)
   
    # 起動シェルを出力 (yolo v2)
    with open(outDir + "start_v2.sh", "w", newline="\n",encoding="utf-8") as wf:
        wf.write("./darknet detector train " +  outDataPath + " "  +outV2CfgPath + " darknet19_448.conv.23")
 
    # 起動シェルを出力 (yolo v3)
    with open(outDir + "start_v3.sh", "w", newline="\n",encoding="utf-8") as wf:
        wf.write("./darknet detector train " +  outDataPath + " "  +outV3CfgPath + " darknet53.conv.74 -dont_show")

    # 起動シェルを出力 (yolo v4)
    with open(outDir + "start_v4.sh", "w", newline="\n",encoding="utf-8") as wf:
        wf.write("./darknet detector train " +  outDataPath + " "  +outV4CfgPath + " yolov4.conv.137 -dont_show")



    # ---------- tiny yolo settings ------------
    # cfgファイルを作成 (tiny yolo v2)
    outV2CfgPath = outDir + "v2_tiny.cfg.txt"
    with open("template_v2_tiny.cfg.txt", encoding="utf-8") as rf:
        with open(outV2CfgPath, "w", newline="\n",encoding="utf-8") as wf:
            cfg = rf.read()
            cfg = cfg.replace("FILTERS_NUM", str((5 + nameLength) * 5 ))
            cfg = cfg.replace("CLASS_NUM", str(nameLength))
            wf.write(cfg)

    # cfgファイルを作成 (tiny yolo v3)
    outV3CfgPath = outDir + "v3_tiny.cfg.txt"
    with open("template_v3_tiny.cfg.txt", encoding="utf-8") as rf:
        with open(outV3CfgPath, "w", newline="\n",encoding="utf-8") as wf:
            cfg = rf.read()
            cfg = cfg.replace("FILTERS_NUM", str((5 + nameLength) * 3 ))
            cfg = cfg.replace("CLASS_NUM", str(nameLength))
            wf.write(cfg)

    # cfgファイルを作成 (tiny yolo v4)
    outV4CfgPath = outDir + "v4_tiny.cfg.txt"
    with open("template_v4_tiny.cfg.txt", encoding="utf-8") as rf:
        with open(outV4CfgPath, "w", newline="\n",encoding="utf-8") as wf:
            cfg = rf.read()
            cfg = cfg.replace("FILTERS_NUM", str((5 + nameLength) * 3 ))
            cfg = cfg.replace("CLASS_NUM", str(nameLength))
            wf.write(cfg)
 
    # 学習済みモデル保存先フォルダ作成 (tiny yolo)
    modelSaveDir = outDir + "tiny_model_saved"
    if not os.path.exists(modelSaveDir):
        os.makedirs(modelSaveDir)

    # dataファイルを作成 (tiny yolo)
    outDataPath = outDir + "tiny_data.txt"
    with open("template_data.txt", encoding="utf-8") as rf:
         with open(outDataPath, "w", newline="\n",encoding="utf-8") as wf:
            data = rf.read()
            data = data.replace("CLASS_NUM", str(nameLength))
            data = data.replace("DATA_LIST_PATH", outDataListPath )
            data = data.replace("NAME_LIST_PATH", outNameFilePath)
            data = data.replace("BACKUP_DIR", modelSaveDir)
            wf.write(data)
   
    # 起動シェルを出力 (tiny yolo v2)
    with open(outDir + "start_tiny_v2.sh", "w", newline="\n",encoding="utf-8") as wf:
        wf.write("./darknet detector train " +  outDataPath + " "  +outV2CfgPath + " darknet19_448.conv.23")

    # 起動シェルを出力 (tiny yolo v3)
    with open(outDir + "start_tiny_v3.sh", "w", newline="\n",encoding="utf-8") as wf:
        wf.write("./darknet detector train " +  outDataPath + " "  +outV3CfgPath + " yolov3-tiny.conv.11 -dont_show")

    # 起動シェルを出力 (tiny yolo v4)
    with open(outDir + "start_tiny_v4.sh", "w", newline="\n",encoding="utf-8") as wf:
        wf.write("./darknet detector train " +  outDataPath + " "  +outV4CfgPath + " yolov4-tiny.conv.29 -dont_show")

