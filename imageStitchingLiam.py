import cv2 as cv
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113


def parse_command_line_arguments():# Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kp", default="SIFT", help="key point (or corner) detector: GFTT ORB SIFT ")
    parser.add_argument("-n", "--nbKp", default=None, type=int, help="Number of key point required (if configurable) ")
    parser.add_argument("-d", "--descriptor", default=True, type=bool, help="compute descriptor associated with detector (if available)")
    parser.add_argument("-m", "--matching", default="NORM_L1", help="Brute Force norm: NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2")
    parser.add_argument("-i1", "--image1", default="IMG_6792 - Petite.jpeg", help="path to image1")
    parser.add_argument("-i2", "--image2", default="IMG_6979 - Petite.jpeg", help="path to image2")
    # other argument may need to be added
    return parser

def test_load_image(img):
    if img is None or img.size == 0 or (img.shape[0] == 0) or (img.shape[1] == 0):
        print("Could not load image !")
        print("Exiting now...")
        exit(1)

def load_gray_image(path):
    if(path != None):
        img = cv.imread(path)
        test_load_image(img)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    else:
        img = None
        gray = None
    return img, gray

def display_image(img, image_window_name):
    cv.namedWindow(image_window_name)
    cv.imshow(image_window_name, img)

def feature_detector(type, gray, nb):
    if gray is not None :
        match type :
            case "GFTT":
                corners = cv.goodFeaturesToTrack(gray, maxCorners=nb, qualityLevel=0.01, minDistance=10)
                kp = []
                if corners is not None:
                    for c in corners:
                        x, y = c.ravel()
                        kp.append(cv.KeyPoint(float(x), float(y), 3.0)) 
            case "ORB":
                orb = cv.ORB_create(nb)
                kp=orb.detect(gray, None)
            case _:
                sift = cv.SIFT_create(nb)
                kp=sift.detect(gray, None)
    else:
        kp =  None
    return kp

def get_norm(norm_name):

    if norm_name == "NORM_L1":
        return cv.NORM_L1
    elif norm_name == "NORM_L2":
        return cv.NORM_L2
    elif norm_name == "NORM_HAMMING":
        return cv.NORM_HAMMING
    elif norm_name == "NORM_HAMMING2":
        return cv.NORM_HAMMING2

def match_descriptors(norm_name, desc1, desc2, alpha=5.0):

    norm = get_norm(norm_name)

    # Brute Force matcher
    bf = cv.BFMatcher(normType=norm, crossCheck=True)

    # Liste de tous les matches
    matches = bf.match(desc1, desc2)

    # On récupère la plus petite distance
    distances = [m.distance for m in matches]
    min_dist = min(distances)

    # Seuil comme dans l’énoncé : seuil = alpha * minDist
    seuil = alpha * min_dist

    # Meilleurs matches
    best_matches = [m for m in matches if m.distance <= seuil]

    print(f"Total matches : {len(matches)} | minDist = {min_dist:.2f} | seuil = {seuil:.2f} | best = {len(best_matches)}")

    return best_matches


def feature_extractor(type, img, kp):
    
    desc = None
    if type == "SIFT":
        sift = cv.SIFT_create()
        kp, desc = sift.compute(img, kp)

    elif type == "ORB":
        orb = cv.ORB_create()
        kp, desc = orb.compute(img, kp)
    
    return desc

def extract_matched_points(kp1, kp2, matches):

    pts1 = []
    pts2 = []

    for m in matches:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1).reshape(-1, 1, 2)
    pts2 = np.float32(pts2).reshape(-1, 1, 2)

    return pts1, pts2

def compute_homography(kp1, kp2, matches):
    if len(matches) < 4:
        print("[ERROR] Pas assez de correspondances pour calculer une homographie.")
        return None, None

    pts1, pts2 = extract_matched_points(kp1, kp2, matches)

    # On veut : pts1 ≈ H * pts2  (H : img2 -> img1)
    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)

    if H is not None:
        inliers = mask.ravel().sum()
        print("Homographie H =\n", H)
        print(f"Inliers RANSAC : {inliers}/{len(matches)}")
    else:
        print("[ERROR] Impossible de calculer l’homographie.")

    return H, mask


def stitch_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Coins de img2
    corners_img2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 1, 2)

    # Coins de img2 après homographie (dans le repère de img1)
    warped_corners_img2 = cv.perspectiveTransform(corners_img2, H)

    # Coins de img1
    corners_img1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    # Tous les coins pour calculer la bounding box
    all_corners = np.concatenate((corners_img1, warped_corners_img2), axis=0)

    xs = all_corners[:, 0, 0]
    ys = all_corners[:, 0, 1]

    min_x = np.floor(xs.min())
    min_y = np.floor(ys.min())
    max_x = np.ceil(xs.max())
    max_y = np.ceil(ys.max())

    # décalage pour avoir des coordonnées positives
    shift_x = int(-min_x if min_x < 0 else 0)
    shift_y = int(-min_y if min_y < 0 else 0)

    # matrice de translation
    T = np.array([
        [1, 0, shift_x],
        [0, 1, shift_y],
        [0, 0, 1]
    ], dtype=np.float32)

    H_shifted = T @ H

    pano_width = int(max_x - min_x)
    pano_height = int(max_y - min_y)

    # On projette img2 dans le grand panorama
    panorama = cv.warpPerspective(img2, H_shifted, (pano_width, pano_height))

    # On colle img1 au bon endroit
    panorama[shift_y:shift_y + h1, shift_x:shift_x + w1] = img1

    return panorama



# other functions will need to be defined

def main():

    parser = parse_command_line_arguments()
    args = vars(parser.parse_args())

    # Load, transform to gray the 2 input images
    print("load image 1")
    img1, gray1 = load_gray_image(args["image1"])
    print("load image 2")
    img2, gray2 = load_gray_image(args["image2"])

    # displays the 2 input images
    #if img1 is not None : display_image(img1, "Image 1")
    #if img2 is not None : display_image(img2, "Image 2")

    # Apply the choosen feature detector
    print(args["kp"]+" detector")
    
    kp1 = feature_detector(args["kp"], gray1, args["nbKp"])
    if img2 is not None: kp2 = feature_detector(args["kp"], gray2, args["nbKp"])

    # Display the keyPoint on the input images
    #img_kp1=cv.drawKeypoints(gray1,kp1,img1)
    #if img2 is not None: img_kp2=cv.drawKeypoints(gray2,kp2,img2)
    
    #display_image(img_kp1, "Image 1 "+args["kp"])
    #if img2 is not None : display_image(img_kp2, "Image 2 "+args["kp"])

    
    # code to complete (using functions):
    # - to extract feature and compute descriptor with ORB and SIFT 
    # Q2 
    desc1 = feature_extractor(args["kp"], gray1, kp1)
    if img2 is not None:
        desc2 = feature_extractor(args["kp"], gray2, kp2)

    # - to calculate brute force matching between descriptor using different norms
    #Q3
    best_matches = []
    if img2 is not None and desc1 is not None and desc2 is not None:
        best_matches = match_descriptors(args["matching"], desc1, desc2, alpha=5.0)
        #Q4
        match_img = cv.drawMatches(
            img1, kp1,
            img2, kp2,
            best_matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        display_image(match_img, f"Matching {args['kp']} {args['matching']}")
    

    # - to calculate and apply homography 
    #Q5
    H = None
    if img2 is not None and len(best_matches) >= 4:
        H, mask = compute_homography(kp1, kp2, best_matches)
    
    # - to stich and display resulting image
    #Q6
    if H is not None:
        panorama = stitch_images(img1, img2, H)
        display_image(panorama, "Panorama")

    # waiting for user action
    key = 0
    while key != ESC_KEY and key!= Q_KEY :
        key = cv.waitKey(1)

    # Destroying all OpenCV windows
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()