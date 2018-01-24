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

def getCurvature(ploty,left_fit,right_fit,leftx,rightx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    return(left_curverad,right_curverad)
    # Example values: 632.1 m    626.2 m


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

def perspectiveTransform(img):
    img = mpimg.imread('.\\output_images\\362_stacked.jpg')   
    src_vertices1 = np.array([[(232,720),(596,435), (678,435), (1200,720)]], dtype=np.int32)  
    cur_imgMask = region_of_interest(img, src_vertices1)    
    cv2.imwrite(".\\output_jpjimages\\"+str(frameCnt)+"_stacked.jpg",cur_imgMask)

    src_vertices = np.array([[(580, 460), (205, 720), (1110, 720), (703, 460)]],dtype=np.float32) 
    dst_vertices = np.array([[(320, 0), (320, 720), (960, 720), (960, 0)]],dtype=np.float32)   

    M = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
    Minv = cv2.getPerspectiveTransform(dst_vertices, src_vertices)

    return(M,Minv)
    
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

# Define a function to threshold an image for a given range and Sobel kernel
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
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255
    
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

    return sxbinary, abs_sobelx, abs_sobely, s_binary

def processFrame(image):
    global frameCnt
    global prv
    global prv5L
    global prv5R
    global current_left_fit
    global current_right_fit
    global previous
    global prv_left_fit
    global prv_right_fit
    global prv_left_fitx
    global prv_right_fitx

    #undistort the image
    dst = cv2.undistort(image,mtx, dist, None, mtx) 
    
    #find the magnitude of the gradient & HLS 
    mag_binary, sobel_absX, sobel_absY, color_bin = magnitude_thresh(dst, sobel_kernel=3, mag_thresh=(30, 150), s_thresh=(170, 255))
    
    #find the direction of the gradient
    dir_binary = direction_threshold(sobel_absX,sobel_absY,thresh=(0.7,1.3))

    combined_MagDir = np.zeros_like(mag_binary)
    combined_MagDir[((mag_binary == 255) & (dir_binary == 255))] = 255
    combined = combined_MagDir + color_bin

    #src_vertices1 = np.array([[(232,720),(596,435), (678,435), (1200,720)]], dtype=np.int32)  
    src_vertices1 = np.array([[(180,720),(600,435), (800,435), (1200,720)]], dtype=np.int32)  

    cur_imgMask = region_of_interest(combined, src_vertices1)    

    warped = cv2.warpPerspective(cur_imgMask, M, (1280, 720),flags=cv2.INTER_LINEAR)

    src_vertices1 = np.array([[(110,720),(110,0), (1200,0), (1200,720)]], dtype=np.int32)  
    warped_reduced = region_of_interest(warped, src_vertices1)    

    rmv_vertices1 = np.array([[(420,720),(590,0), (800,0), (940,720)]], dtype=np.int32)  
    rmv_mask = region_of_interest(warped_reduced, rmv_vertices1) 

    middle_removed = warped_reduced - rmv_mask
#    cv2.imwrite(".\\output_images\\fromVideo\\temp_"+str(frameCnt)+"combined.jpg",combined)
#    cv2.imwrite(".\\output_images\\fromVideo\\temp_"+str(frameCnt)+"cur_imgMas.jpg",cur_imgMask)
#    cv2.imwrite(".\\output_images\\fromVideo\\temp_"+str(frameCnt)+"warped.jpg",warped)
#    cv2.imwrite(".\\output_images\\fromVideo\\temp_"+str(frameCnt)+"warpedreduced.jpg",warped_reduced)
#    cv2.imwrite(".\\output_images\\fromVideo\\temp_"+str(frameCnt)+"rmvmask.jpg",rmv_mask)   
#    cv2.imwrite(".\\output_images\\fromVideo\\temp_"+str(frameCnt)+"middleremoved.jpg",middle_removed)
#    frameCnt += 1
    
    window_centroids = find_window_centroids(middle_removed, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:
    
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(middle_removed)
        r_points = np.zeros_like(middle_removed)
    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = window_mask(window_width,window_height,middle_removed,window_centroids[level][0],level)
    	    r_mask = window_mask(window_width,window_height,middle_removed,window_centroids[level][1],level)
    	    # Add graphic points from window mask here to total pixels found 
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        left_point_indices = np.argwhere(l_points>0)
        right_point_indices = np.argwhere(r_points>0)
    
        ploty = np.linspace(0, middle_removed.shape[0]-1, middle_removed.shape[0] )
        
        # get x and y vectors        
        l_y = left_point_indices[:,0]
        l_x = left_point_indices[:,1]
        # get x and y vectors
        r_y = right_point_indices[:,0]
        r_x = right_point_indices[:,1]
    
        left_fit = np.polyfit(l_y, l_x, 2)
        right_fit = np.polyfit(r_y, r_x, 2)
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]            
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        lcurve, rcurve = getCurvature(ploty,left_fit,right_fit,left_fitx,right_fitx)
        
        if not (lcurve<300):                                          
            if(prv == 10):
                prv = 0
                prv5L[:] = []
                prv5R[:] = []
                prv5L.append([])
                prv5R.append([])
                prv5L[prv].append(left_fit)
                prv5R[prv].append(right_fit)
    #            current_left_fit = np.mean(prv5L, axis=0)
    #            current_right_fit = np.mean(prv5R, axis=0)
                prv += 1
            else:
                prv5L.append([])
                prv5R.append([])
                prv5L[prv].append(left_fit)
                prv5R[prv].append(right_fit)          
                prv += 1
                
            current_left_fit = np.transpose(np.mean(prv5L, axis=0))
            current_right_fit = np.transpose(np.mean(prv5R, axis=0))
            
            current_left_fit = current_left_fit.reshape((3))
            current_right_fit = current_right_fit.reshape((3))
            
            left_fitx = current_left_fit[0]*ploty**2 + current_left_fit[1]*ploty + current_left_fit[2]
            right_fitx = current_right_fit[0]*ploty**2 + current_right_fit[1]*ploty + current_right_fit[2]     
            
            prv_left_fit = current_left_fit
            prv_right_fit = current_right_fit
            prv_left_fitx = left_fitx
            prv_right_fitx = right_fitx
            
        else:
            left_fit = prv_left_fit
            right_fit = prv_right_fit
            left_fitx = prv_left_fitx
            right_fitx = prv_right_fitx
        
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(middle_removed).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*current_left_fit[0]*y_eval + current_left_fit[1])**2)**1.5) / np.absolute(2*current_left_fit[0])
        right_curverad = ((1 + (2*current_right_fit[0]*y_eval + current_right_fit[1])**2)**1.5) / np.absolute(2*current_right_fit[0])
        
        #print(left_curverad, right_curverad)
        # Example values: 1926.74 1908.48
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        left_fit_m = np.polyfit(l_y*ym_per_pix, l_x*xm_per_pix, 2)
        right_fit_m = np.polyfit(r_y*ym_per_pix, r_x*xm_per_pix, 2)

        # Calculate vehicle center
        Max_x = image.shape[1]*xm_per_pix
        Max_y = image.shape[0]*ym_per_pix
        center_of_vehicle = Max_x / 2
        
        Left = left_fit_m[0]*Max_y**2 + left_fit_m[1]*Max_y + left_fit_m[2]
        Right = right_fit_m[0]*Max_y**2 + right_fit_m[1]*Max_y + right_fit_m[2]
        Middle = Left + (Right - Left)/2
        dist_vehicle = Middle - center_of_vehicle
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm', right_curverad, 'm')
        cv2.putText(image,'Left curvature: {:.0f} m'.format(left_curverad), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.putText(image,'Right curvature: {:.0f} m'.format(right_curverad), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        if dist_vehicle>0:
            cv2.putText(image,'Vehicle Position: {:.2f} m to right'.format(dist_vehicle), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        else:
            cv2.putText(image,'Vehicle Position: {:.2f} m to left'.format(-dist_vehicle), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        # Example values: 632.1 m    626.2 m
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result   

#calibrate the camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera('.\\camera_cal\\*.jpg')
frameCnt = 0
# load the video and process frame by frame
undist_output = 'output_images/project_video_undistorted.mp4'
clip2 = VideoFileClip('project_video.mp4')
yellow_clip = clip2.fl_image(processFrame, apply_to=[])
yellow_clip.write_videofile(undist_output, audio=False)
