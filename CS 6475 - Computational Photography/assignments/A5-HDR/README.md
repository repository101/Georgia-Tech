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

The docstrings of each function contains detailed instructions. You are *strongly* encouraged to write your own unit tests based on the requirements. The `test_hdr.py` file is provided to get you started. Your code will be evaluated on input and output type (e.g., uint8, float, etc.), array shape, and values. (Be careful regarding arithmetic overflow!) When you are ready to submit your code, you can send it to the autograder for scoring, but remember that you will only be allowed to submit three times every two (2) hours. In other words, do *not* try to use the autograder as your test suite.

*Notes*:
- Images in the `images/source/sample` directory are provided for testing -- *do not include these images in your submission* (although the output should appear in your report).

- Downsampling your images to 1-2 MB each will save processing time during development. (Larger images take longer to process, and may cause problems for the VM and autograder which are resource-limited.)

- It is essential to put your images in exposure order and name them in this order, similar to the input/sample images. For the given sample images of the home, the exposure info is given in main.py and repeated here (darkest to lightest):
`EXPOSURE TIMES = np.float64([1/160.0, 1/125.0, 1/80.0, 1/60.0, 1/40.0, 1/15.0])`

- Image alignment is critical for HDR. Whether you take your own images or use a set from the web, ensure that your images are aligned and cropped to the same dimensions. This will require a tripod or improvised support. You may use a commercial program such as Gimp (free) or Photoshop ($$) to help with this step. Note what you do in your report.


### 2. Use these functions on your own input images to make an HDR image - *READ CAREFULLY*

You MUST use **at least 5 input images** to create your final HDR image. In most cases, using more input images is better. We recommend resizing images to small versions (like the ones provided) so that the code runs more quickly.

Look online for a set of HDR images with exposure data (EXIF) for exposure times, apertures, and ISO settings. You will need to enter the exposure times (not exposure values) into the main.py code in the same order as the image files are numbered to get a correct HDR result. Go from darkest to lightest. Your results can be amazingly bad if you don't follow this rule.

You may take your own images if your camera supports manual exposure control (at least 5 input images). In particular, the exposure times are required, and aperture and ISO must be reported. Aperture & ISO settings should be held constant. Dark indoor scenes with bright outdoors generally work great for this, or other scenes with overly-bright and dark areas.

**You are REQUIRED to submit the original images that have the required EXIF data we ask for (exposure time, aperture, and ISO).** If the original input images take up too much space in resources.zip, please do the following:
   - Submit the downsampled versions of the input images in resources.zip (submitting input images in resources.zip is **REQUIRED**). If downsampling the images does not keep the EXIF data intact, put the **ORIGINAL** images in a folder on a secure site (ex: Dropbox) and include a working link in your report. Make sure that we can check the EXIF data of these images. Please make sure that your link works and permissions are set, otherwise points will be deducted.


### 3. Above & Beyond

- Taking your own images instead of (or in addition to) using sets from the web will count towards above and beyond credit for this assignment.

- Tone mapping - mapping the high contrast range of the HDR image to fit the limited contrast range of a display or print medium - is the final step in computing an HDR image. Tone mapping is responsible for many of the vibrant colors that are seen when using commercial HDR software. The provided source code does not perform tone mapping (except normalization), so you may experiment with implementing tone mapping on your own (which will count towards above and beyond credit for this assignment). You may use the computeHDR function and add additional functions for your tone mapping efforts. Include detailed information and results comparing your tone mapping to HDR results from the basic code. It can be important for some tone mapping algorithms to know that the output of computeHDR() is the logarithm of the radiance.

Keep in mind:
- Earning the full 10% for A&B is typically _very_ rare; you should not expect to reach it unless your results are _very_ impressive.
- Attempting something very technically difficult does not ensure more credit; make sure you document your effort even if it doesn't pan out.
- Attempting something very easy in a very complicated way does not ensure more credit.


### 4. Complete the report

Make a copy of the [report template](https://drive.google.com/file/d/1vgpfTVBVMpLAORS1X9PoEk9-5Zyi8KC_/view?usp=sharing) and answer all of the questions. Save your report as `report.pdf` in the project directory.


### 5. Submit the Code

**Note:** Make sure that you have completed all the steps in [A0 - Course Setup](../A0-Course_Setup/README.md) for installing the conda environment & other course tools first.

Follow the [Project Submission Instructions](../A0-Course_Setup/README.md#3-submitting-projects) to upload your code `hdr.py` to [Bonnie](https://bonnie.udacity.com) using the `omscs` CLI.


### 6. Submit the Report

Save your report as `report.pdf`. Create an archive named `resources.zip` containing your images and final artifact. Your images must be one of the following types: jpg, jpeg, bmp, png, tif, or tiff.

**Note:** Your `resources.zip` **MUST** include the input images you chose (name `input_1`, `input_2`, etc.), and your HDR result for both the example input set and the input set of your choice (name `exampleResult`, `finalResult`). Again, no need to include the example input images. Submit your `report.pdf` & `resources.zip` **SEPARATELY** on Canvas. **DO NOT** zip `report.pdf` and `resources.zip` together. It is possible that Canvas will automatically rename the files if you resubmit, and it will have a different name when the TAs download it for grading. Do not worry about this if it happens. 

**How to submit Above & Beyond work:** You are allowed to include your A&B work (input images, final results, code files) in `resources.zip` as well. You may name your A&B files how you want, as long as it's clear they are your A&B work (ex: `input1_AB.png`, `result_AB.png`). If there is not enough space (i.e. if you're about to go over the required size limit (see below)), you may put your A&B images in a folder on a secure site (ex: Dropbox) and include a working link in your report.  Keep in mind that for A&B, we cannot grade what we cannot see, so if you state in your report that you implemented something yourself, but do not show us the code, you will not be awarded additional points. Make sure you submit your A&B code files in `resources.zip`.

**Submission Size:** The total size of your project (report.pdf + resources.zip) **MUST** be less than **12MB** for this project. If your submission is too large, you can reduce the scale of your images or report. You can compress your report using [Smallpdf](https://smallpdf.com/compress-pdf).

Example of Canvas submission hierarchy:
   - `report.pdf`
   - `resources.zip`
      - Input image 1 (ex: `input_1.jpg`)
      - Input image 2 (ex: `input_2.jpg`)
      - Input image 3 (ex: `input_3.jpg`)
      - Input image 4 (ex: `input_4.jpg`)
      - Input image 5 (ex: `input_5.jpg`)
      - Final HDR image for example input set (ex: `exampleResult.jpg`)
      - Final HDR image for input set of your choice (ex: `finalResult.jpg`)
      
**Notes:**

  - When sharing images, make sure there is no data contained in the EXIF data that you do not want shared (i.e. GPS). If there is, make sure you strip it out before submitting your work or sharing your photos with others. Normally, we only require that your submitted images include aperture, shutter speed, and ISO (these 3 settings are required for A5).

  - **DO NOT USE 7zip.** We've had problems in the past with 7z archives, so please don't use them unless you don't mind getting a zero on the assignment.


## Criteria for Evaluation

Your submission will be graded based on:

  - Correctness of required code
  - Creativity & overall quality of results
  - Completeness and quality of report
