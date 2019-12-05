import cv2
import numpy as np
import math
import _utils
import argparse
import os.path
import glob
import os
from tqdm import tqdm
import timeit
import sys

# A simple convolution function that returns the filtered images.



@_utils.stop_watch
def getFilterImages(filters, img):
    """
    ガボールフィルタを画像にかけて，出力の画像を返す・．

    input
    -----
    filters: list
    ガボールフィルタが格納されたリスト
    img: ndarray
    入力画像

    output
    -----
    featureaImages: list
    フィルタ適用後の画像のリスト
    """
    featureImages = []
    for filter in filters:
        kern, params = filter
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        featureImages.append(fimg)
    return featureImages

# Apply the R^2 threshold technique here, note we find energy in the spatial domain.
# TODO: このへんがよくわからない！
@_utils.stop_watch
def filterSelection(featureImages, threshold, img, howManyFilterImages):
    """
    エネルギー計算をして，画像の特徴を捉えたフィルターを計算する．

    input
    -----
    featureImages: list
    フィルタをかけた後の画像群

    threshold: 
    論文中では0.95．
    これを超えるフィルタ画像のサブセットが最もよく画像の特徴を捉えているとされている．
    img: 
    入力画像
    howManyFilterImages: int
    何枚の画像を選ぶか．デフォルトで100
    """
    idEnergyList = []
    id = 0
    height, width = img.shape
    for featureImage in featureImages:
        thisEnergy = 0.0
        for x in range(height):
            for y in range(width):
                thisEnergy += pow(np.abs(featureImage[x][y]), 2)
        idEnergyList.append((thisEnergy, id))
        id += 1
    E = 0.0
    for E_i in  idEnergyList:
        E += E_i[0]
    sortedlist = sorted(idEnergyList, key=lambda energy: energy[0], reverse = True)

    tempSum = 0.0
    RSquared = 0.0
    added = 0
    outputFeatureImages = []
    while ((RSquared < threshold) and (added < howManyFilterImages)):
        tempSum += sortedlist[added][0]
        RSquared = (tempSum/E)
        outputFeatureImages.append(featureImages[sortedlist[added][1]])
        added += 1
    return outputFeatureImages

# This is where we create the gabor kernel
# Feel free to uncomment the other list of theta values for testing.
@_utils.stop_watch
def build_filters(lambdas, ksize, gammaSigmaPsi):
    """

    input
    -----
    lambdas: 

    ksize: int
    フィルタのサイズ

    gammaSigmaPsi: list
    ガボールフィルタのパラメタ

    output
    -----
    filters: list
    ガボールフィルタが格納されたリスト
    """

    filters = []
    thetas = []

    # Thetas 1
    # -------------------------------------
    thetas.extend([0, 45, 90, 135])

    # Thetas2
    # -------------------------------------
    #thetas.extend([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])

    thetasInRadians = [np.deg2rad(x) for x in thetas]

    for lamb in lambdas:
        for theta in thetasInRadians:
            params = {'ksize': (ksize, ksize), 'sigma': gammaSigmaPsi[1], 'theta': theta, 'lambd': lamb,
                   'gamma':gammaSigmaPsi[0], 'psi': gammaSigmaPsi[2], 'ktype': cv2.CV_64F}
            kern = cv2.getGaborKernel(**params)
            kern /= 1.5 * kern.sum()
            filters.append((kern, params))
    return filters

# Here is where we convert radial frequencies to wavelengths.
# Feel free to uncomment the other list of lambda values for testing.
@_utils.stop_watch
def getLambdaValues(img):
    """
    入力画像からwavelengthを決める．
    input
    ------
    img: 入力画像

    output
    -----
    lambdaVals: list
    - λの値
    """
    height, width = img.shape

    #calculate radial frequencies.
    max = (width/4) * math.sqrt(2)
    min = 4 * math.sqrt(2)
    temp = min
    radialFrequencies = []

    # Lambda 1
    # -------------------------------------
    while(temp < max):
        radialFrequencies.append(temp)
        temp = temp * 2

    # Lambda 2
    # -------------------------------------
    # while(temp < max):
    #     radialFrequencies.append(temp)
    #     temp = temp * 1.5

    radialFrequencies.append(max)
    lambdaVals = []
    for freq in radialFrequencies:
        lambdaVals.append(width/freq)
    return lambdaVals

# The activation function with gaussian smoothing
@_utils.stop_watch
def nonLinearTransducer(img, gaborImages, L, sigmaWeight, filters):
    """
    画像の非線形変換を行う関数．

    input
    -----
    img: 
    入力画像
    gaborImages: list
    選択後のガボールフィルタ画像

    """

    alpha_ = 0.25
    featureImages = []
    count = 0
    for gaborImage in gaborImages:

        # Spatial method of removing the DC component
        avgPerRow = np.average(gaborImage, axis=0)
        avg = np.average(avgPerRow, axis=0)
        gaborImage = gaborImage.astype(float) - avg

        #gaborImage = cv2.normalize(gaborImage, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Normalization sets the input to the active range [-2,2] this becomes [-8,8] with alpha_
        gaborImage = cv2.normalize(gaborImage, gaborImage, alpha=-8, beta=8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        height, width = gaborImage.shape
        copy = np.zeros(img.shape)
        for row in range(height):
            for col in range(width):
                #centralPixelTangentCalculation_bruteForce(gaborImage, copy, row, col, alpha, L)
                copy[row][col] = math.fabs(math.tanh(alpha_ * (gaborImage[row][col])))

        # now apply smoothing
        copy, destroyImage = applyGaussian(copy, L, sigmaWeight, filters[count])
        if(not destroyImage):
            featureImages.append(copy)
        count += 1

    return featureImages

# I implemented this just for completeness
# It just applies the tanh function and smoothing as spatial convolution
def centralPixelTangentCalculation_bruteForce(img, copy, row, col, alpha, L):
    height, width = img.shape
    windowHeight, windowWidth, inita, initb = \
        _utils.getRanges_for_window_with_adjust(row, col, height, width, L)

    sum = 0.0
    for a in range(windowHeight + 1):
        for b in range(windowWidth + 1):
            truea = inita + a
            trueb = initb + b
            sum += math.fabs(math.tanh(alpha * (img[truea][trueb])))

    copy[row][col] = sum/pow(L, 2)

# Apply Gaussian with the central frequency specification
# @_utils.stop_watch
def applyGaussian(gaborImage, L, sigmaWeight, filter):

    height, N_c = gaborImage.shape

    nparr = np.array(filter[0])
    u_0 = nparr.mean(axis=0)
    u_0 = u_0.mean(axis=0)

    destroyImage = False
    sig = 1
    if (u_0 < 0.000001):
        print('div by zero occured for calculation:')
        print("sigma = sigma_weight * (N_c/u_0), sigma will be set to zero")
        print("removing potential feature image!")
        destroyImage = True
    else:
        sig = sigmaWeight * (N_c / u_0)

    return cv2.GaussianBlur(gaborImage, (L, L), sig), destroyImage

# Remove feature images with variance lower than 0.0001
@_utils.stop_watch
def removeFeatureImagesWithSmallVariance(featureImages, threshold):
    toReturn =[]
    for image in featureImages:
        if(np.var(image) > threshold):
            toReturn.append(image)

    return toReturn



# Our main driver function to return the segmentation of the input image.
@_utils.stop_watch
def runGabor(infile, outfile, k, gk, M, **args):

    # infile = args.infile
    if(not os.path.isfile(infile)):
        print(infile, ' is not a file!')
        exit(0)

    # outfile = args.outfile
    printlocation = os.path.dirname(outfile)
    _utils.deleteExistingSubResults(printlocation)

    M_transducerWindowSize = M
    if((M_transducerWindowSize % 2) == 0):
        # print('Gaussian window size not odd, using next odd number')
        M_transducerWindowSize += 1
    # クラスタ数．何個に画像を分割するか
    k_clusters = k
    # ガボールフィルタのサイズ
    k_gaborSize = gk

    # 各種引数設定
    spatialWeight = args['spw']
    gammaSigmaPsi = []
    gammaSigmaPsi.append(args['gamma'])
    gammaSigmaPsi.append(args['sigma'])
    gammaSigmaPsi.append(args['psi'])
    variance_Threshold = args['vt']
    howManyFeatureImages = args['fi']
    R_threshold = args['R']
    sigmaWeight = args['siw']
    greyOutput = args['c']
    printIntermediateResults = args['i'] 

    # 画像の読み込み
    img = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
    # lambdaの取得.サンプリングレートを決める．
    lambdas = getLambdaValues(img)
    # ガボールフィルタの作成
    filters = build_filters(lambdas, k_gaborSize, gammaSigmaPsi)

    # print("Gabor kernels created, getting filtered images")

    # ガボールフィルタを入力画像にかける．
    filteredImages = getFilterImages(filters, img)
    # どのフィルタを利用するか選ぶ
    filteredImages = filterSelection(filteredImages, R_threshold, img, howManyFeatureImages)

    if(printIntermediateResults):
        _utils.printFeatureImages(filteredImages, "filter", printlocation, infile)

    # print("Applying nonlinear transduction with Gaussian smoothing")

    featureImages = nonLinearTransducer(img, filteredImages, M_transducerWindowSize, sigmaWeight, filters)
    # 分散の小さいデータを除く．
    featureImages = removeFeatureImagesWithSmallVariance(featureImages, variance_Threshold)


    if (printIntermediateResults):
        _utils.printFeatureImages(featureImages, "feature", printlocation, infile)

    # 特徴ベクトルの作成．
    featureVectors = _utils.constructFeatureVectors(featureImages, img)
    # 特徴ベクトルの保存
    featureVectors = _utils.normalizeData(featureVectors, False, spatialWeight=spatialWeight)
    _utils.printFeatureVectors(printlocation, infile, featureVectors)
    
    # print("Clustering...")
    labels = _utils.clusterFeatureVectors(featureVectors, k_clusters)
    _utils.printClassifiedImage(labels, k_clusters, img, outfile, greyOutput)

# For running the program on the command line
def main():
    args = {'spw': 1, 'gamma': 1, 'sigma': 1, 'psi': 0, 'vt': 0.001, 'fi': 100, 'R':0.95, 'siw':0.5, 'c': False, 'i': True}
    IMG_ROOT = "../../ARC_DATAS_RESIZE/experiment_20191205/input"
    k_list = [2, 3]
    gk_list = [32, 64, 128, 256, 512]
    M_list = [5, 10, 15, 20, 25, 30, 35]
    for k in k_list:
        for gk in gk_list:
            for M in M_list:            
                saveFolderName = "k_{}_gk_{}_M_{}".format(str(k), str(gk), str(M))
                print(saveFolderName)
                SAVE_ROOT = "../../ARC_DATAS_RESIZE/experiment_20191205/{}".format(saveFolderName)
                img_path_list = sorted(glob.glob(IMG_ROOT + "/*.jpg"))
                # csv_path = os.path.join(SAVE_ROOT, "{}func_ETA.csv".format(saveFolderName))

                for img_path in img_path_list:
                    # foldername = os.path.basename(os.path.dirname(img_path))
                    filename = os.path.basename(img_path)[:-4]
                    print(filename)
                    save_dir = os.path.join(SAVE_ROOT)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, filename + "_result.jpg")
                    runGabor(img_path, save_path, k, gk, M, **args)
if __name__ == "__main__": 
    # print("AAAA")
    main()