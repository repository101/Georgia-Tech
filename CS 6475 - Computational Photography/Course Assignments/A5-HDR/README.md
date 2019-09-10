# High Dynamic Range Imaging

**Important Note:** This assignment is subject to the "Above & Beyond" rule. In summary: meeting all stated requirements will earn 90%; the last 10% is reserved for individual effort to research, implement, and report some additional high-quality work on this topic beyond the minimum requirements. Your A&B work must be accompanied by discussion of the computational photographic concepts involved, and will be graded based on the level of effort, and quality of your results and documentation in the report. (Please review the full explanation of this rule in the syllabus or on Piazza.)


## Synopsis

In this homework assignment, we will focus on the core algorithms behind computing HDR images based on the paper “Recovering High Dynamic Range Radiance Maps from Photographs” by Debevec & Malik (available in Canvas under Files --> Papers). *It is very important that you read the paper before starting the project.* This project requires a lot of linear algebra. The notational conventions & overall structure is explained in the paper.


## Instructions

### 1. Implement the functions in the `hdr.py` file.

- `linearWeight`: Determine the weight of a pixel based on its intensity.
- `sampleIntensities`: Randomly sample pixel intensity exposure slices for each possible pixel intensity value from the exposure stack.
- `computeResponseCurve`: Find the camera response curve for a single color channel by finding the least-squares solution to an overdetermined system of equations.
- `computeRadianceMap`: Use the response curve to calculate the radiance map for each pixel in the current color layer.

The docstrings of each function contains detailed instructions. You are *strongly* encouraged to write your own unit tests based on the requirements. The `test_hdr.py` file is provided to get you started. Your code will be evaluated on input and output type (e.g., uint8, float, etc.), array shape, and values. (Be careful regarding arithmetic overflow!) When you are ready to submit your code, you can send it to the autograder for scoring by running `omscs submit code` from the root directory of the project folder. Remember that you will only be allowed to submit three times every two (2) hours. In other words, do *not* try to use the autograder as your test suite.

*Notes*:
- Images in the `images/source/sample` directory are provided for testing -- *do not include these images in your submission* (although the output should appear in your report).

- Downsampling your images to 1-2 MB each will save processing time during development. (Larger images take longer to process, and may cause problems for the VM and autograder which are resource-limited.)

- It is essential to put your images in exposure order and name them in this order, similar to the input/sample images. For the given sample images of the home, the exposure info is given in main.py and repeated here (darkest to lightest):
`EXPOSURE TIMES = np.float64([1/160.0, 1/125.0, 1/80.0, 1/60.0, 1/40.0, 1/15.0])`

- Image alignment is critical for HDR. Whether you take your own images or use a set from the web, ensure that your images are aligned and cropped to the same dimensions. This will require a tripod or improvised support. You may use a commercial program such as Gimp (free) or Photoshop ($$) to help with this step. Note what you do in your report.


### 2. Use these function on your own input images to make an HDR image - *READ CAREFULLY*

You MUST use at least 5 input images to create your final HDR image. In most cases, using more input images is better. We recommend resizing images to small versions (like the ones provided) so that the code runs more quickly.

Look online for a set of HDR images with exposure data (EXIF) for exposure times, apertures, and ISO settings. You will need to enter the exposure times (not exposure values) into the main.py code in the same order as the image files are numbered to get a correct HDR result. Go from darkest to lightest. Your results can be amazingly bad if you don't follow this rule.

You may take your own images if your camera supports manual exposure control (at least 5 input images). In particular, the exposure times are required, and aperture and ISO must be reported. Aperture & ISO two settings should be held constant. Dark indoor scenes with bright outdoors generally work great for this, or other scenes with overly-bright and dark areas.


### 3. Above & Beyond

- Taking your own images instead of (or in addition to) using sets from the web will count towards above and beyond credit for this assignment.

- Tone mapping - mapping the high contrast range of the HDR image to fit the limited contrast range of a display or print medium - is the final step in computing an HDR image. Tone mapping is responsible for many of the vibrant colors that are seen when using commercial HDR software. The provided source code does not perform tone mapping (except normalization), so you may experiment with implementing tone mapping on your own (which will count towards above and beyond credit for this assignment). You may use the computeHDR function and add additional functions for your tone mapping efforts. Include detailed information and results comparing your tone mapping to HDR results from the basic code. It can be important for some tone mapping algorithms to know that the output of computeHDR() is the logarithm of the radiance.

Keep in mind:
- Earning the full 10% for A&B is typically _very_ rare; you should not expect to reach it unless your results are _very_ impressive.
- Attempting something very technically difficult does not ensure more credit; make sure you document your effort even if it doesn't pan out.
- Attempting something very easy in a very complicated way does not ensure more credit.


### 4. Complete the report

Make a copy of the [report template](https://docs.google.com/presentation/d/18bceoI6jvtOigp5Hs6z28YaJHynAHDhTEsgvYVclGZs/edit?usp=sharing) and answer all of the questions. Save your report as `report.pdf` in the project directory.


### 5. Submit the Code

**Note:** Make sure that you have completed all the steps in the [instructions](../README.md#virtual-machine-setup) for installing the VM & other course tools first.

Follow the [Project Submission Instructions](../README.md#submitting-projects) to upload your code to [Bonnie](https://bonnie.udacity.com) using the `omscs` CLI:

```
$ omscs submit code
```


### 6. Submit the Report

Save your report as `report.pdf`. Create an archive named `resources.zip` containing your images and final artifact -- both files must be submitted. Your images must be one of the following types: jpg, jpeg, bmp, png, tif, or tiff.

Combine your `report.pdf` & `resources.zip` into a single zip archive and submit the file via Canvas. You may choose any name you want for the combined zip archive, e.g., `assignment6.zip`. Canvas will automatically rename the file if you resubmit, and it will have a different name when the TAs download it for grading. (In other words, you only need to follow the required naming convention for `report.pdf` and `resources.zip` inside your submission archive; don't worry about the name for the combined archive.) YOUR REPORT SUBMISSION FOR THIS PROJECT DOES NOT NEED TO INCLUDE THE CODE, WHICH MUST BE SEPARATELY SUBMITTED TO BONNIE FOR SCORING.

**Note:** The total size of your project (report + resources) must be less than 12MB for this project. If your submission is too large, you can reduce the scale of your images or report. You can compress your report using [Smallpdf](https://smallpdf.com/compress-pdf).

**Note:** Your resources.zip must include your source images and your final HDR image, as well as all images relevant to your A&B. If there isn't enough space, you may put your A&B images in a folder on a secure site (Dropbox, Google Drive, or similar) and include a working link in your report. Again, this only applies to your A&B images. Images that are part of the base assignment should be in resources.zip.


## Criteria for Evaluation

Your submission will be graded based on:

  - Correctness of required code
  - Creativity & overall quality of results
  - Completeness and quality of report
