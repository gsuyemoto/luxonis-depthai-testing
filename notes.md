Recording of video files:
1. Using DepthAI API, so far, the following encoding works:
  a. H265 at 4K (see depthai-experiments/gen2-record-replay)
  b. MJPEG at 4K
  c. color.mp4 file created using video-controller app doesn't seem to work
2. All codecs can be viewed using ffplay or vlc

Stitching using Luxonis hardware:
1. Most likely can use hardware node Warp, although Warp only takes a warp mesh as input and not a homography
2. More information about a warp mesh is located at:
  a. Link: [depthai docs discussion](https://github.com/luxonis/depthai-docs-website/issues/316)
    > Mesh step is what it's set in config, default 16.
    > Does it use all 14 of the distortion coeffs?
    > we use only 8.
    > Is it functionally the same as cv::initUndistortRectifyMap()? 
    > yes
    > loadMeshData overrides useHomographyRectification
  b. Link: [github feature request](https://github.com/luxonis/depthai-core/issues/623)
    > assumes that
    > width = originalImageWidth / meshStep + 1
    > height = originalImageHeight / meshStep + 1
3. Tried stitching where sift/homography is only computed once and then homography is used for all subsequent warp calls. This worked well and retains FPS.
