"""
cat /etc/ld.so.preload 2>/dev/null || echo "no ld.so.preload"
readlink -f /lib/x86_64-linux-gnu/libpthread.so.0
dpkg -S /lib/x86_64-linux-gnu/libpthread.so.0
ldconfig -p | grep libpthread
sudo ldconfig
ldconfig -p | grep libpthread
unset GTK_PATH
unset LD_LIBRARY_PATH
unset GIO_MODULE_DIR

jsp trop



python3 imageStitching.py -k SIFT -n 500 -m NORM_L2
python3 imageStitching.py -k SIFT -n 500 -m NORM_L2 -i1 SEC1c2.jpg -i2 SEC2c2.jpg


"""

import cv2 as cv
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113


def parse_command_line_arguments():  # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-k",
        "--kp",
        default="SIFT",
        help="key point (or corner) detector: GFTT ORB SIFT ",
    )
    parser.add_argument(
        "-n",
        "--nbKp",
        default=None,
        type=int,
        help="Number of key point required (if configurable) ",
    )
    parser.add_argument(
        "-d",
        "--descriptor",
        default=True,
        type=bool,
        help="compute descriptor associated with detector (if available)",
    )
    parser.add_argument(
        "-m",
        "--matching",
        default="NORM_L1",
        help="Brute Force norm: NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2",
    )
    parser.add_argument(
        "-i1", "--image1", default="./IMG_1_reduced.jpg", help="path to image1"
    )
    parser.add_argument(
        "-i2", "--image2", default="./IMG_2_reduced.jpg", help="path to image2"
    )
    # other argument may need to be added
    return parser


def test_load_image(img):
    if img is None or img.size == 0 or (img.shape[0] == 0) or (img.shape[1] == 0):
        print("Could not load image !")
        print("Exiting now...")
        exit(1)


def load_gray_image(path):
    if path != None:
        img = cv.imread(path)
        test_load_image(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img = None
        gray = None
    return img, gray


def display_image(img, image_window_name):
    cv.namedWindow(image_window_name)
    cv.imshow(image_window_name, img)


def feature_detector(detector_type, gray, nb):
    if gray is None:
        return None
    if nb is None:
        nb = 500
    t = detector_type.upper()
    if t == "ORB":
        orb = cv.ORB_create(nfeatures=nb)
        keypoints = orb.detect(gray, None)
    else:  # SIFT par défaut
        sift = cv.SIFT_create(nb)
        keypoints = sift.detect(gray, None)

    return keypoints


def feature_extractor(detector_type, gray_image, keypoints):
    t = detector_type.upper()

    if t == "ORB":
        extractor = cv.ORB_create()
    else: 
        extractor = cv.SIFT_create()

    keypoints, descriptors = extractor.compute(gray_image, keypoints)
    return keypoints, descriptors


def match_descriptors(norm_type_str, desc1, desc2, alpha=4.0):
    if desc1 is None or desc2 is None:
        return [], []

    if norm_type_str == "NORM_L1":
        norm_type = cv.NORM_L1
    elif norm_type_str == "NORM_L2":
        norm_type = cv.NORM_L2
    elif norm_type_str == "NORM_HAMMING":
        norm_type = cv.NORM_HAMMING
    elif norm_type_str == "NORM_HAMMING2":
        norm_type = cv.NORM_HAMMING2
    else:
        norm_type = cv.NORM_L2

    bf = cv.BFMatcher(norm_type, crossCheck=False)

    matches = bf.match(desc1, desc2)

    if len(matches) == 0:
        return [], []

    matches = sorted(matches, key=lambda m: m.distance)

    min_dist = matches[0].distance
    seuil = alpha * min_dist

    best_matches = [m for m in matches if m.distance < seuil]

    return matches, best_matches


def homo(best_matches, kp1, kp2):
    src_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return H, mask


def stitch_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    panoramique_width = w1 + w2
    panoramique_height = max(h1, h2)
    panoramique = cv.warpPerspective(img2, H, (panoramique_width, panoramique_height))
    panoramique[0:h1, 0:w1] = img1

    return panoramique


def display_resized(img, window_name, max_width=1200, max_height=800):
    h, w = img.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)  # 1.0 = jamais agrandir

    new_size = (int(w * scale), int(h * scale))
    resized = cv.resize(img, new_size, interpolation=cv.INTER_AREA)

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, resized)


def main():

    parser = parse_command_line_arguments()
    args = vars(parser.parse_args())

    # Load, transform to gray the 2 input images
    print("load image 1")
    img1, gray1 = load_gray_image(args["image1"])
    print("load image 2")
    img2, gray2 = load_gray_image(args["image2"])

    """# displays the 2 input images
    if img1 is not None:
        display_image(img1, "Image 1")
    if img2 is not None:
        display_image(img2, "Image 2")
    print("debug1")"""
    # Apply the choosen feature detector
    print(args["kp"] + " detector")

    kp1 = feature_detector(args["kp"], gray1, args["nbKp"])
    if img2 is not None:
        kp2 = feature_detector(args["kp"], gray2, args["nbKp"])

    # Display the keyPoint on the input images
    img_kp1 = cv.drawKeypoints(gray1, kp1, img1.copy())
    img_kp2 = cv.drawKeypoints(gray2, kp2, img2.copy())

    """
    display_image(img_kp1, "Image 1 " + args["kp"])
    display_image(img_kp2, "Image 2 " + args["kp"])
    """
    # code to complete (using functions):
    # - to extract feature and compute descriptor with ORB and SIFT

    kp1, desc1 = feature_extractor(args["kp"], gray1, kp1)
    kp2, desc2 = feature_extractor(args["kp"], gray2, kp2)

    print("Descripteurs image 1 :", desc1.shape)
    print("Descripteurs image 2 :", desc2.shape)
    print("\nimage1")
    print("nb kp:", len(kp1))
    print("1er kp:", kp1[0].pt)
    print("1er desc:", desc1[0][:8])  # 8 valeurs, brut

    print("\nimage2")
    print("nb kp:", len(kp2))
    print("1er kp:", kp2[0].pt)
    print("1er desc:", desc2[0][:8])  # 8 valeurs, brut

    # - to calculate brute force matching between descriptor using different norms

    matches, best_matches = match_descriptors(args["matching"], desc1, desc2, alpha=4.0)

    print("\nmatching :")
    print("nb matches totaux :", len(matches))
    print("nb meilleurs matches :", len(best_matches))


    img_matches = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        best_matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    #display_image(img_matches, "Meilleurs matchs")
    display_resized(img_matches, "Meilleurs matchs")
    

    # - to calculate and apply homography
    H, mask = homo(best_matches, kp1, kp2)

    if H is not None:
        print("\nHomographie H :")
        print(H)
        print("nb inliers (RANSAC) :", int(mask.sum()))

        # - to stich and display resulting image

        panorama = stitch_images(img1, img2, H)
        #display_image(panorama, "Panorama")
        display_resized(panorama, "Panorama")
    

    # waiting for user action
    key = 0
    while key != ESC_KEY and key != Q_KEY:
        key = cv.waitKey(1)

    # Destroying all OpenCV windows
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
