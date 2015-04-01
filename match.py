# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
    help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
    help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

# load the image image, convert it to grayscale, and detect edges
orignal_template = cv2.imread(args["template"])
bgr2Gray_template = cv2.cvtColor(orignal_template, cv2.COLOR_BGR2GRAY)
gaussianBlur_Template = cv2.GaussianBlur(bgr2Gray_template, (5, 5), 0)
canny_Template = cv2.Canny(gaussianBlur_Template, 50, 200)
origin_canny_Template = cv2.Canny(bgr2Gray_template, 50, 200)

# template3 = cv2.Canny(template2, 50, 200)

(tH, tW) = orignal_template.shape[:2]

# cv2.imshow("blurred", template3)
# cv2.imshow("originan canned", template)

cv2.imshow("canny_Template", canny_Template)
cv2.imshow("origin_canny_Template", origin_canny_Template)
cv2.waitKey(0)


# loop over the images to find the template in
for imagePath in glob.glob(args["images"] + "/*.png"):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    print '-----debug1--------'
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        print '-----debug2--------'
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        resized = cv2.GaussianBlur(resized, (5, 5), 0)
        edged = cv2.Canny(resized, 50, 200)

        # cv2.imshow("resized", edged)
        # cv2.waitKey(0)

        #Methods: TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF_NORMED, TM_CCOEFF
        result = cv2.matchTemplate(edged, canny_Template, cv2.TM_CCORR)
        (min_val, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if args.get("visualize", False):
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)

        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            print 'didnt find any match'

    # unpack the bookkeeping varaible and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found


    print min_val
    print maxVal
    if abs(min_val) > -950000000:
      print "it is a good match!"
      (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
      print (startX, startY)
      (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

      print '-----debug3--------'
      # draw a bounding box around the detected result and display the image
      cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
      cv2.imshow("Image", image)
      cv2.waitKey(0)
      


#   