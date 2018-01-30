# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:41:24 2018

@author: Ira
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

M = np.array([[-0.50772,-1.49582,951.33],
              [-3.21965e-15,-1.98816,914.554],
              [-4.98733e-18,-0.00238604,1]
              ])

Minv = np.array([[0.192187,-0.766859,518.5],
                 [1.77636e-15,-0.502977,460],
                 [-1.73472e-18,-0.00120012,1]
                 ])

prv5L = []
prv5R = []
current_left_fit = []
current_right_fit = []
prv = 0
previous = 0
prv_left_fit = []
prv_right_fit = []
prv_left_fitx = []
prv_right_fitx = []

def direction_threshold(sobelx, sobely,  thresh=(0, np.pi/2)):

    absgraddir = np.arctan2(sobely, sobelx)
    absgraddir_degree = (absgraddir / np.pi) * 180
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir_degree >= 40) & (absgraddir_degree <= 75)] = 255

    # Return the binary image
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def magnitude_thresh(img, sobel_kernel=3, mag_thresh=(0, 255),  s_thresh=(170, 255)):
    
    # 1) Convert to grayscale
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0,sobel_kernel)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1,sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)    
    abs_sobelxy= np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))    
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    sxbinary_x = np.zeros_like(scaled_sobelx)
    sxbinary_y = np.zeros_like(scaled_sobely)    
    sxbinary_x[(scaled_sobelx >= mag_thresh[0]) & (scaled_sobelx <= mag_thresh[1])] = 255
    sxbinary_y[(scaled_sobely >= mag_thresh[0]) & (scaled_sobely <= mag_thresh[1])] = 255
    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 255

    return sxbinary, abs_sobelx, abs_sobely

def getCurvature(ploty,left_fit,right_fit,leftx,rightx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / \
    np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / \
    np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    return(left_curverad,right_curverad)
    # Example values: 632.1 m    626.2 m

   
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def calibrate_camera(Image_Path):
    global counter
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objpoints = []
    imgpoints = []
    
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    images = glob.glob(Image_Path) #
    
    counter = 0
    
    for fname in images:
       img = cv2.imread(fname)
       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
       # Find the chess board corners
       ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
       
       # If found, add object points, image points (after refining them)
       if ret == True:
           objpoints.append(objp)
        
           cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
           imgpoints.append(corners)
        
           # Draw and display the corners
           cv2.drawChessboardCorners(img, (9,6), corners,ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, gray.shape[::-1],None, None)
    return ret, mtx, dist, rvecs, tvecs

def perspectiveTransform(img):

    src_vertices = np.array([[(587, 446), (153, 673), (1126, 673), (691, 446)]],dtype=np.float32) 
    dst_vertices = np.array([[(200, 0), (200, 720), (1080, 720), (1080, 0)]],dtype=np.float32)   

    M = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
    Minv = cv2.getPerspectiveTransform(dst_vertices, src_vertices)

    return(M,Minv)
    
def hls_mask(img):

    white_lwr = np.array([0, 210, 0])
    white_upr = np.array([255, 255, 255])
    
    yellow_lwr = np.array([20, 0, 100])
    yellow_upr = np.array([30, 220, 255])
    
    # Convert the scale from RGB to HLS
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # white color mask
    white_mask = cv2.inRange(hls_img, white_lwr, white_upr)


    # yellow color mask
    yellow_mask = cv2.inRange(hls_img, yellow_lwr, yellow_upr)

    return white_mask, yellow_mask


def processFrame(image):
    global start
    global prv_left_fit 
    global prv_right_fit
    global prv_curvature
 
    y_eval = 700 #np.max(ploty)
    midx = 640
    xm_per_pix = 3.7/660.0 # meters per pixel in x dimension
    ym_per_pix = 30/720 # meters per pixel in y dimension
    nwindows = 9
    margin = 100
    minpix = 50

    #undistort the image
    dst = cv2.undistort(image,mtx, dist, None, mtx) 

    #find the magnitude of the gradient 
    mag_binary, sobel_absX, sobel_absY = magnitude_thresh(dst, \
                                                          sobel_kernel=3, \
                                                          mag_thresh=(30, 150), \
                                                          s_thresh=(170, 255))
    
    #find the direction of the gradient
    dir_binary = direction_threshold(sobel_absX,sobel_absY,thresh=(0.7,1.3))

    combined_MagDir = np.zeros_like(mag_binary)
    combined_MagDir[((mag_binary == 255) & (dir_binary == 255))] = 255  
     
    w_color, y_color = hls_mask(dst)
    
    combined = np.zeros_like(w_color)
    combined[((w_color == 255) | (y_color == 255))] = 255
    combined[(combined == 255)] = 255

#    temp = np.zeros_like(w_color)
#    temp[((combined == 255)|(combined_MagDir== 255))] = 255
#    
#    combined = temp
    
    warped = cv2.warpPerspective(combined, M, (1280, 720),flags=cv2.INTER_LINEAR)
    
    window_height = np.int(warped.shape[0]/nwindows)
    
    if start:
        histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)

        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & \
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & \
                               (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        prv_right_fit = right_fit
        prv_left_fit = left_fit
        
        y1 = (2*right_fit[0]*y_eval + right_fit[1])*xm_per_pix/ym_per_pix
        y2 = 2*right_fit[0]*xm_per_pix/(ym_per_pix**2)
        curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
        
        if (curvature) < 500:
            prv_curvature = 0.75*curvature + 0.25*(((1 + y1*y1)**(1.5))/np.absolute(y2)) 
        
        start = 0
 
    else:
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = prv_left_fit
        right_fit = prv_right_fit
        
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & \
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & \
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        left_error = ((prv_left_fit[0] - left_fit[0]) ** 2).mean(axis=None)      
        right_error = ((prv_right_fit[0] - right_fit[0]) ** 2).mean(axis=None)        
        if left_error < 0.01:
            prv_left_fit = 0.75 * prv_left_fit + 0.25 * left_fit   
        if right_error < 0.01:
            prv_right_fit = 0.75 * prv_right_fit + 0.25 * right_fit
        
        y1 = (2*right_fit[0]*y_eval + right_fit[1])*xm_per_pix/ym_per_pix
        y2 = 2*right_fit[0]*xm_per_pix/(ym_per_pix**2)
        curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)

        prv_curvature = curvature
              
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    cv2.putText(result,'Radius of Curvature: %.2fm' % curvature,(20,40), \
                cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    
    x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
    x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
        
    position_from_center = ((x_left_pix + x_right_pix)/2.0 - midx) * xm_per_pix

    if position_from_center < 0:
        text = 'left'
    else:
        text = 'right'
    cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), text),(20,80), \
                cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    return result
    


#calibrate the camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera('.\\camera_cal\\*.jpg')

frameCnt = 0
start = 1
prv_left_fit = [np.array([False])] 
prv_right_fit = [np.array([False])] 
prv_curvature = 0

#test_images = glob.glob('.\\test_Images\\*.jpg')
#for fname in test_images:
#    img = mpimg.imread(fname)   
#    temp = fname.split('\\')
#    filename = temp[2].split('.jpg')
#    temp1 ='.\\test_Images\\'+ filename[0]+'_out.jpg'
#    result = processFrame(img)
#    lab = hls_mask(img)
#    cv2.imwrite(temp1,result)


# load the video and process frame by frame
undist_output = 'output_images/project_video_undistorted.mp4'
clip2 = VideoFileClip('project_video.mp4')
yellow_clip = clip2.fl_image(processFrame, apply_to=[])
yellow_clip.write_videofile(undist_output, audio=False)
